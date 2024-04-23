from model.embedder import *
import torch.nn as nn



class RenderingNetworkPEGASUS(nn.Module):
    # RenderingNetworkSceneLatentThreeStages
    def __init__(
            self,
            optimize_scene_latent_code,     # NOTE custom 
            latent_code_dim,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_pnts=0,
    ):
        super().__init__()

        # NOTE custom ###############
        self.optimize_scene_latent_code = optimize_scene_latent_code
        if self.optimize_scene_latent_code:
            self.scene_latent_dim = latent_code_dim
        else:
            self.scene_latent_dim = 0

        # dims = [d_in + feature_vector_size] + dims + [d_out]
        dims = [d_in + feature_vector_size + self.scene_latent_dim] + dims + [d_out]
        self.d_in = d_in
        self.embedview_fn = None
        self.embedpnts_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            # dims[0] += (input_ch - 3)
            dims[0] += (input_ch - 3) + self.scene_latent_dim

        if multires_pnts > 0:
            embedpnts_fn, input_ch_pnts = get_embedder(multires_pnts)
            self.embedpnts_fn = embedpnts_fn
            # dims[0] += (input_ch_pnts - 3)
            dims[0] += (input_ch_pnts - 3) + self.scene_latent_dim

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, normals, condition):
        x = normals

        if self.optimize_scene_latent_code:
            x = torch.cat([x, condition['scene_latent']], dim=1)
            
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x
    