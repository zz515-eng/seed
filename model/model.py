import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RnaClassifier(nn.Module):
    def __init__(self, 
                 embedding_dim=2560, 
                 trinuc_vocab_size=65, 
                 trinuc_embedding_dim=2560,
                 sec_struct_vocab_size=3, 
                 sec_struct_emb_dim=2560,
                 num_classes=2, 
                 nhead=8, 
                 num_encoder_layers=2, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 attention_dropout=0.1,
                 pooling_strategy='mean'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        
        # 1. 独立嵌入层 (针对附加特征)
        # 序列特征已经是 token 级别的密集向量了 (来自 LucaVirus)，所以不需要 Embedding 层
        self.trinucleotide_embedding = nn.Embedding(trinuc_vocab_size, trinuc_embedding_dim)
        self.secondary_structure_embedding = nn.Embedding(sec_struct_vocab_size, sec_struct_emb_dim)
        
        # 组合特征维度
        self.combined_dim = embedding_dim + trinuc_embedding_dim + sec_struct_emb_dim
        
        # --- 新增：降维层 (提速核心) ---
        # 将 7680 维的拼接特征降维到 1024 维，大幅减少 Conformer 的计算量和显存占用
        self.conformer_dim = 1024
        self.feature_projection = nn.Sequential(
            nn.Linear(self.combined_dim, self.conformer_dim),
            nn.LayerNorm(self.conformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dropout层
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 方案 B: SHAPE 投影桥 (映射层)
        # 将 [序列+三联体+结构](1024) + [SHAPE](1) 映射回 1024 维
        self.shape_projection = nn.Linear(self.conformer_dim + 1, self.conformer_dim)
        
        # 池化策略
        self.pooling_strategy = pooling_strategy

        # Conformer 层
        try:
            from conformer import Conformer
            self.conformer = Conformer(
                dim=self.conformer_dim,
                depth=num_encoder_layers,
                heads=nhead,
                ff_mult=dim_feedforward // self.conformer_dim if dim_feedforward > self.conformer_dim else 1,
                conv_expansion_factor=2,
                conv_kernel_size=31,
                attn_dropout=attention_dropout,
                ff_dropout=dropout,
                conv_dropout=dropout
            )
            print("✅ 成功加载 Conformer (lucidrains) 分类器 (降维提速版, dim={}, dropout={})".format(self.conformer_dim, dropout))
        except ImportError:
            # 回退到 torchaudio
            try:
                from torchaudio.models import Conformer
                self.conformer = Conformer(
                    input_dim=self.conformer_dim,
                    num_heads=nhead,
                    ffn_dim=dim_feedforward,
                    num_layers=num_encoder_layers,
                    depthwise_conv_kernel_size=31,
                    dropout=dropout
                )
                self.use_torchaudio_conformer = True
                print("✅ 成功加载 Conformer (torchaudio) 分类器 (降维提速版, dim={}, dropout={})".format(self.conformer_dim, dropout))
            except ImportError:
                raise ImportError("❌ 请安装 conformer 以使用架构: pip install conformer")

        # 6. 分类头 (如果使用均值池化，输出为 [batch_size, conformer_dim])
        self.classifier_head = nn.Sequential(
            nn.Linear(self.conformer_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),                    
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),                    
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)             
        )

    def forward(self, seq_token_embeddings, trinuc_indices, sec_struct_indices, shape_values=None):
        # 1. 获取嵌入并立即应用dropout
        trinuc_emb = self.embedding_dropout(self.trinucleotide_embedding(trinuc_indices))
        sec_struct_emb = self.embedding_dropout(self.secondary_structure_embedding(sec_struct_indices))
        
        # 2. 特征组合：直接拼接所有特征，保持维度 (例如 2560*3 = 7680)
        combined_features = torch.cat((seq_token_embeddings, trinuc_emb, sec_struct_emb), dim=-1)
        
        # --- 新增：通过投影层将 7680 降维到 1024 ---
        combined_features = self.feature_projection(combined_features)

        # 2.5 注入 SHAPE 反应值 (方案 B)
        if shape_values is not None:
            # shape_values 形状预计为 (batch_size, seq_len, 1)
            # 拼接后维度为 conformer_dim + 1
            combined_features = torch.cat((combined_features, shape_values), dim=-1)
            # 投影回 conformer_dim 维以兼容后续主干
            combined_features = self.shape_projection(combined_features)
            combined_features = nn.functional.relu(combined_features)

        # 3. 前向传播 Conformer
        if getattr(self, 'use_torchaudio_conformer', False):
            # torchaudio 的 conformer 需要 lengths
            batch_size, seq_len, _ = combined_features.shape
            lengths = torch.full((batch_size,), seq_len, device=combined_features.device, dtype=torch.long)
            combined_features, _ = self.conformer(combined_features, lengths)
        else:
            # pip install conformer 直接前向传播
            combined_features = self.conformer(combined_features)

        # 4. 池化
        if self.pooling_strategy == 'mean':
            pooled_features = combined_features.mean(dim=1)
        elif self.pooling_strategy == 'max':
            pooled_features = combined_features.max(dim=1)[0]
        elif self.pooling_strategy == 'cls':
            # 假设第一个 token 是 CLS
            pooled_features = combined_features[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # 5. 分类头
        logits = self.classifier_head(pooled_features)
        
        return logits
