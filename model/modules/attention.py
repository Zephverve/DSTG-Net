# import numpy as np 
# from torch import nn 


# class Attention (nn .Module ):


#     def __init__ (self ,dim_in ,dim_out ,num_heads =8 ,qkv_bias =False ,qk_scale =None ,attn_drop =0. ,proj_drop =0. ,
#     mode ='spatial'):
#         super ().__init__ ()
#         self .num_heads =num_heads 
#         head_dim =dim_in //num_heads 
#         self .scale =qk_scale or head_dim **-0.5 

#         self .attn_drop =nn .Dropout (attn_drop )
#         self .proj =nn .Linear (dim_in ,dim_out )
#         self .mode =mode 
#         self .qkv =nn .Linear (dim_in ,dim_in *3 ,bias =qkv_bias )
#         self .proj_drop =nn .Dropout (proj_drop )

#     def forward (self ,x ):
#         B ,T ,J ,C =x .shape 

#         qkv =self .qkv (x ).reshape (B ,T ,J ,3 ,self .num_heads ,C //self .num_heads ).permute (3 ,0 ,4 ,1 ,2 ,
#         5 )
#         if self .mode =='temporal':
#             q ,k ,v =qkv [0 ],qkv [1 ],qkv [2 ]
#             x =self .forward_temporal (q ,k ,v )
#         elif self .mode =='spatial':
#             q ,k ,v =qkv [0 ],qkv [1 ],qkv [2 ]
#             x =self .forward_spatial (q ,k ,v )
#         else :
#             raise NotImplementedError (self .mode )
#         x =self .proj (x )
#         x =self .proj_drop (x )
#         return x 

#     def forward_spatial (self ,q ,k ,v ):
#         B ,H ,T ,J ,C =q .shape 
#         attn =(q @k .transpose (-2 ,-1 ))*self .scale 
#         attn =attn .softmax (dim =-1 )
#         attn =self .attn_drop (attn )

#         x =attn @v 
#         x =x .permute (0 ,2 ,3 ,1 ,4 ).reshape (B ,T ,J ,C *self .num_heads )

#         return x 

#     def forward_temporal (self ,q ,k ,v ):
#         B ,H ,T ,J ,C =q .shape 
#         qt =q .transpose (2 ,3 )
#         kt =k .transpose (2 ,3 )
#         vt =v .transpose (2 ,3 )

#         attn =(qt @kt .transpose (-2 ,-1 ))*self .scale 
#         attn =attn .softmax (dim =-1 )
#         attn =self .attn_drop (attn )

#         x =attn @vt 
#         x =x .permute (0 ,3 ,2 ,1 ,4 ).reshape (B ,T ,J ,C *self .num_heads )

#         return x 










from torch import nn
import torch

class Attention(nn.Module):
    """
    FlashAttention2-compatible attention.
    Input: (B, T, J, C)
    Mode: 'spatial' or 'temporal'
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., mode='spatial'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.mode = mode

        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_in, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # detect flash-attn
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash = True
            print(f"✅ FlashAttention2 enabled for {mode}-attention")
        except Exception as e:
            print(f"⚠️ FlashAttention2 not available ({e}), fallback to softmax attention.")
            self.flash_attn_func = None
            self.use_flash = False

    def forward(self, x):
        B, T, J, C = x.shape
        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, H, T, J, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.mode == 'temporal':
            x = self.forward_temporal(q, k, v)
        elif self.mode == 'spatial':
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # ---- spatial attention (seq = J) ----
    def forward_spatial(self, q, k, v):
        B, H, T, J, D = q.shape
        q = q.reshape(B * T, H, J, D)
        k = k.reshape(B * T, H, J, D)
        v = v.reshape(B * T, H, J, D)

        if self.use_flash:
            # FlashAttention2 expects [B', S, H, D]
            out = self.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       dropout_p=0.0, causal=False)
            out = out.transpose(1, 2).contiguous()
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.reshape(B, T, J, H * D)
        return out

    # ---- temporal attention (seq = T) ----
    def forward_temporal(self, q, k, v):
        B, H, T, J, D = q.shape
        # swap T <-> J to make temporal sequence
        q = q.permute(0, 1, 3, 2, 4).reshape(B * J, H, T, D)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * J, H, T, D)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * J, H, T, D)

        if self.use_flash:
            out = self.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       dropout_p=0.0, causal=False)
            out = out.transpose(1, 2).contiguous()
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        out = out.reshape(B, J, T, H * D).permute(0, 2, 1, 3)
        return out














































































































































































