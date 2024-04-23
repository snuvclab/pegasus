import torch
from model.embedder import *
import numpy as np
import torch.nn as nn
from functorch import vmap


class GeometryNetworkPEGASUS(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            optimize_latent_code,
            optimize_scene_latent_code,
            latent_code_dim,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
    ):
        super().__init__()

        self.optimize_latent_code = optimize_latent_code
        self.optimize_scene_latent_code = optimize_scene_latent_code

        if self.optimize_latent_code:
            self.latent_dim = latent_code_dim
        else:
            self.latent_dim = 0
        if self.optimize_scene_latent_code:
            self.scene_latent_dim = latent_code_dim
        else:
            self.scene_latent_dim = 0

        # dims = [d_in] + dims + [d_out + feature_vector_size]
        dims = [d_in + self.latent_dim + self.scene_latent_dim] + dims + [d_out + feature_vector_size]

        self.feature_vector_size = feature_vector_size
        self.embed_fn = None
        self.multires = multires
        self.bias = bias
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            # dims[0] = input_ch
            dims[0] = input_ch + self.latent_dim + self.scene_latent_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        
        # NOTE semantic network
        seg_dims = [512, 1]
        self.seg_num_layers = len(seg_dims)
        for l in range(0, self.seg_num_layers - 1):
            out_dim = seg_dims[l + 1]

            lin = nn.Linear(seg_dims[l], out_dim)

            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            torch.nn.init.constant_(lin.bias, 0.0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin_seg" + str(l), lin)

    def forward(self, input, condition, grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input) # 800, 3 -> 800, 39
        
        # hyunsoo added
        if self.optimize_latent_code: # [24576, 32]가 맞는 상황인가?
            # Currently only support batch_size=1
            # This is because the current implementation of masking in ray tracing doesn't support other batch sizes.
            num_pixels = int(input.shape[0] / condition['latent'].shape[0])
            condition_latent = condition['latent'].unsqueeze(1).expand(-1, num_pixels, -1).reshape(-1, self.latent_dim)
            input = torch.cat([input, condition_latent], dim=1)

        if self.optimize_scene_latent_code:
            if grad:
                input = torch.cat([input, condition['scene_latent_gradient']], dim=1)
            else:
                input = torch.cat([input, condition['scene_latent']], dim=1)

        x = input # 6400, 71

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l == self.num_layers - self.seg_num_layers - 1:
                seg_x = x

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        # NOTE binary classification
        for l in range(0, self.seg_num_layers - 1):
            lin_seg = getattr(self, "lin_seg" + str(l))

            seg_x = lin_seg(seg_x)
            
            if l < self.seg_num_layers - 2:
                seg_x = self.softplus(seg_x)

        x = torch.cat([x, seg_x], dim=1)

        return x 

    def gradient(self, x, condition):
        x.requires_grad_(True)
        output = self.forward(x, condition, grad=True) # x: 800, 3 I guess this is the point?
        y = output[:,:1]            # NOTE SDF에 해당. 나머지 3dim은 feature_vector이며 albedo로 취급한다.
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(            # NOTE canonical gradient를 구하는 IMAvatar와 코드가 정확히 동일한 것으로 보인다.
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return output, gradients # gradient는 그대로이다. canonical point가 1set이기 때문에 뒤에서 shading할 때 문제가 된다.
    
