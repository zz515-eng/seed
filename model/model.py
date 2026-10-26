"""ADAR FiLM classifier: LucaVirus tokens modulated by trinuc/structure, not concatenated.

Same Conformer dim / readout / classifier head as the concat model.
Default readout is the center position. Pass pooling_strategy="mean" only to
reproduce the older human_film_mean checkpoint.
"""
import torch
import torch.nn as nn


class RnaClassifier(nn.Module):
    def __init__(
        self,
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
        pooling_strategy="center",
        cond_dim=128,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.fusion = "film"
        self.conformer_dim = 1024
        self.cond_dim = cond_dim

        self.trinucleotide_embedding = nn.Embedding(trinuc_vocab_size, trinuc_embedding_dim)
        self.secondary_structure_embedding = nn.Embedding(sec_struct_vocab_size, sec_struct_emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.token_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.conformer_dim),
            nn.LayerNorm(self.conformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # trinuc + structure + 1 SHAPE channel (zeros if missing)
        self.cond_in_dim = trinuc_embedding_dim + sec_struct_emb_dim + 1
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_in_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.film_gen = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.conformer_dim * 2),
        )
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

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
                conv_dropout=dropout,
            )
            print(
                "Loaded Conformer (lucidrains) classifier (FiLM, dim={}, dropout={}, pooling={})".format(
                    self.conformer_dim, dropout, pooling_strategy
                )
            )
        except ImportError:
            try:
                from torchaudio.models import Conformer

                self.conformer = Conformer(
                    input_dim=self.conformer_dim,
                    num_heads=nhead,
                    ffn_dim=dim_feedforward,
                    num_layers=num_encoder_layers,
                    depthwise_conv_kernel_size=31,
                    dropout=dropout,
                )
                self.use_torchaudio_conformer = True
                print(
                    "Loaded Conformer (torchaudio) classifier (FiLM, dim={}, dropout={}, pooling={})".format(
                        self.conformer_dim, dropout, pooling_strategy
                    )
                )
            except ImportError:
                raise ImportError("Install conformer: pip install conformer")

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
            nn.Linear(128, num_classes),
        )

    def forward(self, seq_token_embeddings, trinuc_indices, sec_struct_indices, shape_values=None):
        x = self.token_projection(seq_token_embeddings)
        trinuc_emb = self.embedding_dropout(self.trinucleotide_embedding(trinuc_indices))
        sec_struct_emb = self.embedding_dropout(self.secondary_structure_embedding(sec_struct_indices))
        if shape_values is None:
            shape_values = torch.zeros(
                seq_token_embeddings.size(0),
                seq_token_embeddings.size(1),
                1,
                device=seq_token_embeddings.device,
                dtype=x.dtype,
            )
        elif shape_values.dim() == 2:
            shape_values = shape_values.unsqueeze(-1)
        cond = self.cond_mlp(torch.cat((trinuc_emb, sec_struct_emb, shape_values.to(dtype=x.dtype)), dim=-1))
        gamma, beta = self.film_gen(cond).chunk(2, dim=-1)
        x = x * (1.0 + gamma) + beta

        if getattr(self, "use_torchaudio_conformer", False):
            batch_size, seq_len, _ = x.shape
            lengths = torch.full((batch_size,), seq_len, device=x.device, dtype=torch.long)
            x, _ = self.conformer(x, lengths)
        else:
            x = self.conformer(x)

        if self.pooling_strategy == "mean":
            pooled = x.mean(dim=1)
        elif self.pooling_strategy == "max":
            pooled = x.max(dim=1)[0]
        elif self.pooling_strategy == "cls":
            pooled = x[:, 0, :]
        elif self.pooling_strategy == "center":
            pooled = x[:, x.size(1) // 2, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        return self.classifier_head(pooled)
