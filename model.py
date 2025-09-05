import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SingleHeadAttention(nn.Module):
    def __init__(self, max_seq_len, d_head, mask=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_head = d_head
        self.Wq = nn.Linear(d_head, d_head)
        self.Wk = nn.Linear(d_head, d_head)
        self.Wv = nn.Linear(d_head, d_head)

        if mask:
            self.register_buffer("attn_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool())
        else:
            self.attn_mask = None

    def forward(self, x):  # B, T, C
        B, T, C = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)

        if self.attn_mask is not None:
            attn_scores = attn_scores.masked_fill(self.attn_mask[:T, :T], float("-inf"))

        attn_outputs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_outputs, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, mask=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = d_model // n_heads
        self.heads = nn.ModuleList(
            [SingleHeadAttention(max_seq_len, self.d_head, mask=mask) for _ in range(n_heads)]
        )

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        attn_outputs = []
        for i in range(self.n_heads):
            attn_output = self.heads[i](x[..., i * self.d_head: (i + 1) * self.d_head])
            attn_outputs.append(attn_output)

        attn_concat = torch.cat(attn_outputs, dim=-1)
        output = self.output_proj(attn_concat)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, mask=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.attn = MultiHeadAttention(max_seq_len, d_model, n_heads, mask=mask)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 3*d_model),
            nn.LeakyReLU(0.1),
            nn.Linear(3*d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):
        norm1 = self.ln1(x)
        attn_out = self.attn(norm1)
        x = x + attn_out

        norm2 = self.ln2(x)
        out = self.mlp(norm2)
        return x + out


class TransformerNetwork(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads, n_layers, vocab_size, mask=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(
            *[TransformerLayer(max_seq_len, d_model, n_heads, mask=mask) for _ in range(n_layers)]
        )
        self.vocab_proj = nn.Linear(d_model, vocab_size)

        self.register_buffer("pos_indices", torch.arange(max_seq_len).unsqueeze(dim=0))


    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(self.pos_indices[:, :T])
        output = tok_emb + pos_emb

        for block in self.blocks:
            output = block(output)

        logits = self.vocab_proj(output)
        return logits