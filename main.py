import torch 
import torch.nn as nn
import torch.nn.functional as F
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
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
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
    self.norm1 = nn.LayerNorm(dim)
    self.attn = MultiHeadAttention(dim, num_heads, dropout)
    self.norm2 = nn.LayerNorm(dim)

    mlp_hidden_dim = int(dim* mlp_ratio)
    self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(mlp_hidden_dim, dim),nn.Dropout(dropout))


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

    self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride = patch_size)

    self.time_embed = TimestepEmbedding(hidden_dim)

    self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patches, hidden_dim))

    self.blocks = nn.ModuleList([
      TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
      for _ in range(depth)
    ])

    self.norm = nn.LayerNorm(hidden_dim)
    self.head = nn.Sequential(
      nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
      nn.GELU(),
      nn.ConvTranspose2d(
        hidden_dim,in_channels,kernel_size=patch_size,
        stride = patch_size
      )
    )

    self.initialize_weights()

  def initialize_weights(self):
    w = self.patch_embed.weight.data

    torch.nn.init.xavier_uniform_(w.view([w.shape[0],-1]))

    torch.nn.init.normal_(self.pos_embed, std=0.02)

  def forward(self, x , timesteps):
    x = self.patch_embed(x)
    B, C, H, W = x.shape
    x = x.flatten(2).transpose(1,2)

    x = x + self.pos_embed

    time_embed = self.time_embed(timesteps)

    x = x + time_embed.unsqueeze(1)

    for block in self.blocks:
      x = block(x)

    x = self.norm(x)
    x = x.transpose(1,2).reshape(B, C, H, W)
    x = self.head(x)

    return x
  
def test_dit_model():

  print("start Dit")
  print ("\nshape check")

  model = DiT(input_size=32, patch_size=2,in_channels=3, hidden_dim=384, depth = 12, num_heads=6)

  batch_size =4 
  x = torch.randn(batch_size,3,32,32)
  timesteps = torch.randint(0,100,(batch_size,))

  output = model(x, timesteps)

  assert output.shape == x.shape, f"shape mismatch {x.shape}{output.shape}"
  print("\n shape test pssed")

  print("\nvariable batch size handling")
  batch_size = [1,8,16]
  for bs in batch_size:
    x = torch.randn(bs,3,32,32)
    timesteps = torch.randint(0,1000,(bs,))
    output = model(x, timesteps)
    assert output.shape == x.shape, f"wrong bs {bs}"

  print("\n bs test pass")

  print("\nTimestepEmbeddingg test ")
  time_embed = TimestepEmbedding(384)
  test_times = torch.tensor([0,500,999])
  embeddings = time_embed(test_times)
  assert embeddings.shape == (3,384), f"Failed embeddings {embeddings.shape}"

  emb1 = time_embed(torch.tensor([100]))
  emb2 = time_embed(torch.tensor([200]))
  assert not torch.allclose(emb1, emb2), "diffrent timesteps produced sasme embeddings"

  print("\nTimestep test passed")



  print("\n Multi head attention test")

  attn = MultiHeadAttention(dim=384, num_heads=6)
  x = torch.randn(4,256, 384)
  out = attn(x)
  assert out.shape == x.shape , f"Attention shape mismatch {out.shape} vs {x.shape}"
  print("\nmultihead attn test passed")

  print("transformer test")

  block = TransformerBlock(dim = 384, num_heads=6)

  x = torch.randn(4,256,384)
  out = block(x)
  assert out.shape == x.shape, f"Transformer block shape mismatch {out.shape} vs {x.shape}"
  print("\n transformer test pass")

  return "Done"

if __name__=="__main__":
  test_dit_model()

