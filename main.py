import torch 
import torch.nn as nn
import torch.nn.function as F
import math

class TimestepEmbedding(nn.Module):
  def __init__(self, embedding_dim,max_period=10000):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.max_period = max_period

  def forward(self, timesteps):
    half = self.embedding_dim // 2
    freqs = torch.exp(-math.log(self.max_period) * torch.arange(start=0, end = half,dtype=torch.float32)/half).to(device = timesteps.device)
    args = timesteps[:,None].float() * freqs[None]
    embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.embedding_dim % 2 :
      embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:1])], dim=-1)
    return embedding

class MultiHeadAttention(nn.Module):
  def __init__(self, dim,num_heads=8, dropout=0.1):
    super().__init__()
    assert dim % num_heads ==0, ' Dimentions must be devisible by heads'
    
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(dim , dim * 3)
    self.proj = nn.Linear(dim, dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N , 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)

    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2,-1)) + self.scale
    attn = attn.softmax(dim=-1)
    attn = self.dropout(attn)

    x = (attn @ v).transpose(1,2).reshape(B, N, C)
    x = self.proj(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self, dim, num_heads = 8, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.norml = nn.LayerNorm(dim)
    self.attn = MultiHeadAttention(dim, num_heads, dropout)
    self.norm2 = nn.LayerNorm(dim)

    mlp_hidden_dim = int(dim* mlp_ratio)
    self.mlp == nn.Sequential(
      nn.Linear(dim, mlp_hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(mlp_hidden_dim, dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
  
class DiT(nn.Module):
  def __init__(
      self,
      input_size=32,
      patch_size=2,
      in_channels=3,
      hidden_dim=384,
      depth=12,
      num_heads=6,
      mlp_ratio=4,
      dropout=0.1
  ):
    super().__init__()

    self.input_size = input_size
    self.patch_size = patch_size
    self.hidden_dim = hidden_dim
    self.num_patches = (input_size // patch_size) ** 2

    self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernal_size*patch_size, stride = patch_size)

    self.time_embed = TimestepEmbedding(hidden_dim)

    self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patches, hidden_dim))

    self.blocks = nn.ModuleList([
      TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
      for _ in range(depth)
    ])
