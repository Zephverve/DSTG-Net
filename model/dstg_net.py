from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from timm.models.layers import DropPath

from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP
from model.modules.tcn import MultiScaleTCN


class DSTGBlock(nn.Module):
    """
    DSTG spatial/temporal mixer block.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=20,num_joints=22):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim,
                             num_nodes=num_joints if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("DSTG mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        self.mixer_type = mixer_type
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)


    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            if self.mixer_type == 'graph':
                x = self.mixer(x)
            else:
                x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DSTGNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=20,
                 num_joints=22):
        super().__init__()


        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        self.att_spatial = DSTGBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qkv_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames,num_joints=num_joints)
        self.att_temporal = DSTGBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qkv_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames,num_joints=num_joints)

        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=num_joints,
                                     mode='spatial')
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len)
        else:
            self.graph_spatial = DSTGBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qkv_scale, use_layer_scale, layer_scale_init_value,
                                               mode='spatial', mixer_type="graph",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames,num_joints=num_joints)
            self.graph_temporal = DSTGBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qkv_scale, use_layer_scale, layer_scale_init_value,
                                                mode='temporal', mixer_type="ms-tcn" if use_tcn else 'graph',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames,num_joints=num_joints)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            with torch.amp.autocast('cuda'):
                x_attn_spatial = self.att_spatial(x)
                x_graph_spatial = self.graph_spatial(x)

                x_attn_future = torch.jit.fork(self.att_temporal, x_attn_spatial)
                x_graph_future = torch.jit.fork(self.graph_temporal, x_graph_spatial)
                x_attn = torch.jit.wait(x_attn_future)
                x_graph = torch.jit.wait(x_graph_future)





        z_attn_agg = x_attn.mean(dim=(1, 2))
        z_graph_agg = x_graph.mean(dim=(1, 2))

        z_attn_norm = F.normalize(z_attn_agg, p=2, dim=1)
        z_graph_norm = F.normalize(z_graph_agg, p=2, dim=1)

        decorrelation_loss = F.cosine_similarity(z_attn_norm, z_graph_norm, dim=-1).mean()

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]

        else:
            x = (x_attn + x_graph) * 0.5

        return x,decorrelation_loss


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=20,
                 num_joints=22):
    layers = []
    for _ in range(n_layers):
        layers.append(DSTGNetBlock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop=drop_rate,
            drop_path=drop_path_rate,
            num_heads=num_heads,
            use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion,
            hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn,
            graph_only=graph_only,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
            num_joints=num_joints,
        ))
    return nn.ModuleList(layers)


class DSTGNet(nn.Module):
    def __init__(self, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3, n_layers=16, mlp_ratio=4.,
                 act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None,
                 layer_scale_init_value=1e-5, use_adaptive_fusion=True, hierarchical=False,
                 use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                 graph_only=False, neighbour_num=4, n_frames=20, num_joints=22):
        super().__init__()
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.time_embed = nn.Parameter(torch.randn(1, n_frames, 1, dim_feat))
        self.adj = nn.Parameter(torch.ones(num_joints, num_joints))
        self.adj_temporal = nn.Parameter(torch.ones(n_frames, n_frames))
        self.n_frames = n_frames

        self.layers = create_layers(
            dim=dim_feat,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop_rate=drop,
            drop_path_rate=drop_path,
            num_heads=num_heads,
            use_layer_scale=use_layer_scale,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion,
            hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn,
            graph_only=graph_only,
            neighbour_num=neighbour_num,
            n_frames=n_frames,
            num_joints=num_joints,
        )

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.ReLU()),
        ]))
        self.norm = nn.LayerNorm(dim_feat)
        self.head = nn.Linear(dim_feat, dim_out)

    def forward(self, x1, return_rep=False):
        """
        """
        b, t, j, c = x1.shape
        x = self.joints_embed(x1)
        x = x + self.pos_embed

        total_decorrelation_loss = 0.0
        for layer in self.layers:
            x, decorrelation_loss = layer(x)
            if decorrelation_loss is not None:
                total_decorrelation_loss += decorrelation_loss

        z = self.head(x)

        return z,total_decorrelation_loss/len(self.layers)




def _test():
    from torchprofile import profile_macs
    import warnings
    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 20, 22
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = DSTGNet(n_layers=12, dim_in=3, dim_feat=64, mlp_ratio=4, hierarchical=False,
                           use_tcn=False, graph_only=False, n_frames=t).to('cuda')
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100 
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    average_inference_time = (end_time - start_time) / num_iterations

    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")
    

    out, _ = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"


if __name__ == '__main__':
    _test()
