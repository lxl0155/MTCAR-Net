from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, in_channel, embed_dim, norm_layer, modifysequencelength=False):
        super().__init__()
        self.modifysequencelength = modifysequencelength #
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(nn.Linear(in_channel, embed_dim, bias=False),
                                  nn.GELU(),
                                  nn.Linear(embed_dim, embed_dim, bias=False),
                                  nn.GELU()
                                  )
        if norm_layer is not None:
            if norm_layer == 'nn.LayerNorm':
                self.norm = nn.LayerNorm(embed_dim)
            else:
                self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.flatten(2).transpose(1, 2)  

        if self.modifysequencelength:
            x = self.proj(x.transpose(1, 2)) 
            
        else:
            x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)
        return x
     