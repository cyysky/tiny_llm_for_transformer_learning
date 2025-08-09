import torch
import torch.nn as nn
from torch.nn import functional as F

class TinyConfig:
    def __init__(self,
                 vocab_size=32000,
                 dim=256,
                 n_layers=4,
                 n_heads=4,
                 max_seq_len=512,
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout


class TinyTransformer(nn.Module):
    def __init__(self, config: TinyConfig):
        super().__init__()
        # Add HuggingFace-style config object for PEFT compatibility
        class HFConfig:
            def __init__(self, cfg):
                self.model_type = "custom_tinytransformer"
                self.vocab_size = cfg.vocab_size
                self.hidden_size = cfg.dim
                self.num_hidden_layers = cfg.n_layers
                self.num_attention_heads = cfg.n_heads
                self.max_position_embeddings = cfg.max_seq_len
                self.tie_word_embeddings = False

            def get(self, key, default=None):
                return getattr(self, key, default)

        self.config = HFConfig(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.n_heads,
                dim_feedforward=config.dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        # Align with HF/PEFT expected calling convention
        if input_ids is None and "idx" in kwargs:
            input_ids = kwargs["idx"]
        return {"idx": input_ids}

    def forward(self, idx=None, input_ids=None, targets=None, labels=None, attention_mask=None, **kwargs):
        # Accept both our native (idx, targets) and HF-style (input_ids, labels, attention_mask)
        if input_ids is not None:
            idx = input_ids
        if labels is not None:
            targets = labels
        # attention_mask currently unused in this tiny model, but accepted for compatibility
        return self._forward_internal(idx, targets)

    def _forward_internal(self, idx, targets=None):

        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            return logits, loss
        return logits