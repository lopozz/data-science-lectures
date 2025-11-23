import torch
from torch import nn
from torch.nn import functional as F

class GPT2Model(nn.Module):
    """
    A minimal GPT-2 style decoder-only transformer language model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        vocab_size: int,
        max_position_embeddings: int,
        num_layers: int,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(max_position_embeddings, embed_dim)
        self.blocks = nn.Sequential(
            *[GPT2Block(embed_dim, num_heads, use_cache) for _ in range(num_layers)]
        )

        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(sequence_length, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layernorm(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            _, _, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last allowed token
            idx_cond = idx[:, -self.max_position_embeddings :]
            logits, _ = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class GPT2Block(nn.Module):
    def __init__(self, embed_dim, num_heads, use_cache):
        super().__init__()

        self.attn_layer = GPT2MultiHeadAttention(
            embed_dim, num_heads, use_cache=use_cache
        )
        self.mlp_layer = GPT2MLP(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn_layer(self.layernorm1(x))
        x = x + self.mlp_layer(self.layernorm2(x))

        return x


class GPT2MLP(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )  # see https://arxiv.org/pdf/1706.03762#page=5

    def forward(self, x):
        return self.net(x)


class GPT2MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product for causal attention.

    This module implements the standard multi-head attention mechanism introduced in
    the Transformer architecture (Vaswani et al., 2017). The idea behind multi-head
    attention is to allow the model to jointly attend to information from different
    representation subspaces at different positions. Instead of performing a single
    attention operation on the full embedding dimension, the input is projected into
    several smaller “heads,” each of which performs attention independently. The
    outputs of all heads are then concatenated and linearly projected back to the
    model dimension.
    Multi-head attention increases the representational power of the model by
    splitting the embedding dimension `embed_dim` into `num_heads` smaller subspaces
    (`head_dim = embed_dim // num_heads`)

    References
    ----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
      Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
      https://arxiv.org/abs/1706.03762

    """

    def __init__(
        self, embed_dim: int, num_heads: int, attn_pdrop: float = 0.1, use_cache=False
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.use_cache = use_cache
        self.kv_cache = None

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(attn_pdrop)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (B, nH, T, H)
        k = k.permute(0, 2, 1, 3)  # (B, nH, T, H)
        v = v.permute(0, 2, 1, 3)  # (B, nH, T, H)

        kq = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, nH, T, T)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=kq.device, dtype=torch.bool), diagonal=1
        )  # (T, T)
        kq = kq.masked_fill(mask, float("-inf"))
        att = F.softmax(kq, dim=-1)

        att = self.dropout(att)

        o = att @ v

        o = o.permute(0, 2, 1, 3).contiguous()  # (B, T, nH, H)
        o = o.view(batch_size, seq_len, embed_dim)  # concat heads
        o = self.Wo(o)
        o = self.dropout(o)
        return o
