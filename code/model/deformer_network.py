import torch
from model.embedder import *
import numpy as np
import torch.nn as nn


class ForwardDeformerPEGASUS(nn.Module):
    def __init__(self,
                FLAMEServer,
                d_in,
                dims,
                multires,
                optimize_scene_latent_code,
                latent_code_dim,
                num_exp=50,
                deform_c=False,
                deform_cc=False,
                weight_norm=True,
                ghostbone=False,
                ):
        super().__init__()
        # NOTE custom ######################
        self.optimize_scene_latent_code = optimize_scene_latent_code
        if self.optimize_scene_latent_code:
            self.scene_latent_dim = latent_code_dim
        else:
            self.scene_latent_dim = 0
        ####################################

        self.FLAMEServer = FLAMEServer
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3

        if deform_cc and not deform_c:
            assert False, 'deform_cc should be False when deform_c is False'

        if deform_c:
            d_out = d_out + 3
        if deform_cc:
            # NOTE cano-canonical offset
            d_out = d_out + 128
            
        self.num_exp = num_exp
        self.deform_c = deform_c
        self.deform_cc = deform_cc
        # dims = [d_in] + dims + [d_out]                                                        # NOTE original
        dims = [d_in + self.scene_latent_dim] + dims + [d_out]   # NOTE custom
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            # dims[0] = input_ch                                                        # NOTE original
            dims[0] = input_ch + self.scene_latent_dim   # NOTE custom

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        # NOTE cc custom ########
        if self.deform_cc:
            self.cc_linear = nn.Linear(128, 128)
            self.cc = nn.Linear(128, 3)
        #########################
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)
        # NOTE custom
        if self.deform_cc:
            torch.nn.init.constant_(self.cc_linear.bias, 0.0)
            torch.nn.init.normal_(self.cc_linear.weight, 0.0, np.sqrt(2) / np.sqrt(128))
            torch.nn.init.constant_(self.cc.bias, 0.0)
            torch.nn.init.constant_(self.cc.weight, 0.0)

        self.ghostbone = ghostbone

    def canocanonical_deform(self, pnts_c, mask=None, cond=None):
        if not self.deform_cc:
            return pnts_c

        assert self.deform_cc, 'deform_cc should be True'
        if mask is not None:
            pnts_c = pnts_c[mask]
            # custom #########################
            if self.optimize_scene_latent_code:
                condition_scene_latent = cond['scene_latent'][mask]
            ##################################

        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c
        
        # custom #########################
        if self.optimize_scene_latent_code:
            if mask is None:
                condition_scene_latent = cond['scene_latent']
            assert condition_scene_latent.shape[0] == x.shape[0], 'x dim: {}, condition_scene_latent dim: {}'.format(x.shape, condition_scene_latent.shape)
            x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
        ##################################

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        pnts_c = pnts_c + self.cc(self.cc_linear(blendshapes[:, -131:-3]))     # NOTE offset을 더해준다. paper: canonical offset.
    
        return pnts_c
    
    def query_weights(self, pnts_c, mask=None, cond=None):
        if mask is not None:
            pnts_c = pnts_c[mask]
            # custom #########################
            if self.optimize_scene_latent_code:
                condition_scene_latent = cond['scene_latent'][mask]
            ##################################

        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c
        
        # custom #########################
        if self.optimize_scene_latent_code:
            if mask is None:
                condition_scene_latent = cond['scene_latent']
            assert condition_scene_latent.shape[0] == x.shape[0], 'double check'
            x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
        ##################################

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        # softmax implementation
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
        if self.deform_c:
            pnts_c_flame = pnts_c + blendshapes[:, -3:]     # NOTE offset을 더해준다. paper: canonical offset.
        else:
            pnts_c_flame = pnts_c
        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None, cond=None):
        shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights(pnts_c, mask, cond)
        pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        return pts_p, pnts_c_flame



# class ForwardDeformerSceneLatentDeepThreeStagesBlendingInference(nn.Module):
#     def __init__(self,
#                 FLAMEServer,
#                 d_in,
#                 dims,
#                 multires,
#                 optimize_scene_latent_code,
#                 latent_code_dim,
#                 num_exp=50,
#                 deform_c=False,
#                 deform_cc=False,
#                 weight_norm=True,
#                 ghostbone=False,
#                 ):
#         super().__init__()
#         # NOTE custom ######################
#         self.optimize_scene_latent_code = optimize_scene_latent_code
#         if self.optimize_scene_latent_code:
#             self.scene_latent_dim = latent_code_dim
#         else:
#             self.scene_latent_dim = 0
#         ####################################

#         self.FLAMEServer = FLAMEServer
#         # pose correctives, expression blendshapes and linear blend skinning weights
#         d_out = 36 * 3 + num_exp * 3
#         if deform_c:
#             d_out = d_out + 3
#         if deform_cc:
#             # NOTE cano-canonical offset
#             d_out = d_out + 128
#         self.num_exp = num_exp
#         self.deform_c = deform_c
#         self.deform_cc = deform_cc
#         # dims = [d_in] + dims + [d_out]                                                        # NOTE original
#         dims = [d_in + self.scene_latent_dim] + dims + [d_out]   # NOTE custom
#         self.embed_fn = None
#         if multires > 0:
#             embed_fn, input_ch = get_embedder(multires)
#             self.embed_fn = embed_fn
#             # dims[0] = input_ch                                                        # NOTE original
#             dims[0] = input_ch + self.scene_latent_dim   # NOTE custom

#         self.num_layers = len(dims)
#         for l in range(0, self.num_layers - 2):
#             out_dim = dims[l + 1]
#             lin = nn.Linear(dims[l], out_dim)

#             torch.nn.init.constant_(lin.bias, 0.0)
#             torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "lin" + str(l), lin)

#         self.softplus = nn.Softplus(beta=100)
#         self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
#         # NOTE cc custom ########
#         if deform_cc:
#             self.cc_linear = nn.Linear(128, 128)
#             self.cc = nn.Linear(128, 3)
#         #########################
#         self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
#         self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
#         torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
#         torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
#         if weight_norm:
#             self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
#         # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
#         torch.nn.init.constant_(self.blendshapes.bias, 0.0)
#         torch.nn.init.constant_(self.blendshapes.weight, 0.0)
#         torch.nn.init.constant_(self.skinning.bias, 0.0)
#         torch.nn.init.constant_(self.skinning.weight, 0.0)
#         # NOTE custom
#         if deform_cc:
#             torch.nn.init.constant_(self.cc_linear.bias, 0.0)
#             torch.nn.init.normal_(self.cc_linear.weight, 0.0, np.sqrt(2) / np.sqrt(128))
#             torch.nn.init.constant_(self.cc.bias, 0.0)
#             torch.nn.init.constant_(self.cc.weight, 0.0)

#         self.ghostbone = ghostbone

#     def canocanonical_deform(self, pnts_c, mask=None, cond=None):
#         assert self.deform_cc, 'deform_cc should be True'
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_scene_latent = cond['scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x = self.embed_fn(pnts_c)
#         else:
#             x = pnts_c
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_scene_latent = cond['scene_latent']
#             assert condition_scene_latent.shape[0] == x.shape[0], 'x dim: {}, condition_scene_latent dim: {}'.format(x.shape, condition_scene_latent.shape)
#             x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         pnts_c = pnts_c + self.cc(self.cc_linear(blendshapes[:, -131:-3]))     # NOTE offset을 더해준다. paper: canonical offset.
    
#         return pnts_c
    
#     def query_weights(self, pnts_c, mask=None, cond=None):
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_scene_latent = cond['scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x = self.embed_fn(pnts_c)
#         else:
#             x = pnts_c
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_scene_latent = cond['scene_latent']
#             assert condition_scene_latent.shape[0] == x.shape[0], 'double check'
#             x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         posedirs = blendshapes[:, :36 * 3]
#         shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
#         lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
#         # softmax implementation
#         lbs_weights_exp = torch.exp(20 * lbs_weights)
#         lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
#         if self.deform_c:
#             pnts_c_flame = pnts_c + blendshapes[:, -3:]     # NOTE offset을 더해준다. paper: canonical offset.
#         else:
#             pnts_c_flame = pnts_c
#         return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

#     def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None, cond=None):
#         shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights(pnts_c, mask, cond)
#         pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
#         return pts_p, pnts_c_flame
    
#     def canocanonical_deform_blending(self, pnts_c, mask=None, cond=None, mask_point=None):
#         assert self.deform_cc, 'deform_cc should be True'
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_source_scene_latent = cond['source_scene_latent'][mask]
#                 condition_target_scene_latent = cond['target_scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x = self.embed_fn(pnts_c)
#         else:
#             x = pnts_c
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_source_scene_latent = cond['source_scene_latent']
#                 condition_target_scene_latent = cond['target_scene_latent']

#             # Step 1: Use masks to index into x
#             x_segment = x[mask_point['segment_mask']]                                                   # NOTE [1102, 3]
#             x_unsegment = x[mask_point['unsegment_mask']]                                               # NOTE [107622, 3]

#             # Step 2: Concatenate x with the condition latent tensors
#             concat_segment = torch.cat((x_segment, condition_source_scene_latent), dim=1)               # NOTE [1102, 323]
#             concat_unsegment = torch.cat((x_unsegment, condition_target_scene_latent), dim=1)           # NOTE [107622, 323]

#             # Step 3: Assemble everything into the final tensor
#             final_shape = (x.shape[0], x.shape[1] + condition_source_scene_latent.shape[1])
#             final_tensor = torch.zeros(final_shape, device=pnts_c.device)                               # NOTE [108724, 323]
#             final_tensor.index_add_(0, mask_point['segment_mask'], concat_segment)
#             final_tensor.index_add_(0, mask_point['unsegment_mask'], concat_unsegment)

#             x = torch.cat([x, condition_source_scene_latent], dim=1) # [300000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         pnts_c = pnts_c + self.cc(self.cc_linear(blendshapes[:, -131:-3]))     # NOTE offset을 더해준다. paper: canonical offset.
    
#         return pnts_c
    
#     def query_weights_blending(self, pnts_c_segmented, pnts_c_unsegmented, mask=None, cond=None):
#         if mask is not None:
#             pnts_c_segmented = pnts_c_segmented[mask]
#             pnts_c_unsegmented = pnts_c_unsegmented[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_source_scene_latent = cond['source_scene_latent'][mask]
#                 condition_target_scene_latent = cond['target_scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x_segmented = self.embed_fn(pnts_c_segmented)
#             x_unsegmented = self.embed_fn(pnts_c_unsegmented)
#         else:
#             x_segmented = pnts_c_segmented
#             x_unsegmented = pnts_c_unsegmented
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_source_scene_latent = cond['source_scene_latent']
#                 condition_target_scene_latent = cond['target_scene_latent']
#             assert condition_source_scene_latent.shape[0] == x_segmented.shape[0], 'double check'
#             x_segmented = torch.cat([x_segmented, condition_source_scene_latent], dim=1) # [300000, 71]
#             x_unsegmented = torch.cat([x_unsegmented, condition_target_scene_latent], dim=1) # [300000, 71]
#             x = torch.cat([x_segmented, x_unsegmented], dim=0) # [600000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         posedirs = blendshapes[:, :36 * 3]
#         shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
#         lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
#         # softmax implementation
#         lbs_weights_exp = torch.exp(20 * lbs_weights)
#         lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)

#         pnts_c = torch.cat([pnts_c_segmented, pnts_c_unsegmented], dim=0)
#         if self.deform_c:
#             pnts_c_flame = pnts_c + blendshapes[:, -3:]     # NOTE offset을 더해준다. paper: canonical offset.
#         else:
#             pnts_c_flame = pnts_c
#         return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame

#     def forward_lbs_blending(self, pnts_c_segmented, pnts_c_unsegmented, pose_feature, betas, transformations, mask=None, cond=None):
#         shapedirs, posedirs, lbs_weights, pnts_c_flame = self.query_weights_blending(pnts_c_segmented, pnts_c_unsegmented, mask, cond)
#         pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
#         return pts_p, pnts_c_flame




# class ForwardDeformerSceneLatentDeepThreeStagesShapeCancel(nn.Module):
#     def __init__(self,
#                 FLAMEServer,
#                 d_in,
#                 dims,
#                 multires,
#                 optimize_scene_latent_code,
#                 latent_code_dim,
#                 num_exp=50,
#                 deform_c=False,
#                 deform_cc=False,
#                 weight_norm=True,
#                 ghostbone=False,
#                 ):
#         super().__init__()
#         # NOTE custom ######################
#         self.optimize_scene_latent_code = optimize_scene_latent_code
#         if self.optimize_scene_latent_code:
#             self.scene_latent_dim = latent_code_dim
#         else:
#             self.scene_latent_dim = 0
#         ####################################

#         self.FLAMEServer = FLAMEServer

#         self.num_betashape = 100                                                            # NOTE beta cancel
#         # pose correctives, expression blendshapes and linear blend skinning weights
#         # d_out = 36 * 3 + num_exp * 3
#         d_out = 36 * 3 + num_exp * 3 + self.num_betashape * 3                               # NOTE beta cancel
#         if deform_c:
#             d_out = d_out + 3
#         if deform_cc:
#             # NOTE cano-canonical offset
#             d_out = d_out + 128
#         self.num_exp = num_exp
#         self.deform_c = deform_c
#         self.deform_cc = deform_cc
#         # dims = [d_in] + dims + [d_out]                                                        # NOTE original
#         dims = [d_in + self.scene_latent_dim] + dims + [d_out]   # NOTE custom
#         self.embed_fn = None
#         if multires > 0:
#             embed_fn, input_ch = get_embedder(multires)
#             self.embed_fn = embed_fn
#             # dims[0] = input_ch                                                        # NOTE original
#             dims[0] = input_ch + self.scene_latent_dim   # NOTE custom

#         self.num_layers = len(dims)
#         for l in range(0, self.num_layers - 2):
#             out_dim = dims[l + 1]
#             lin = nn.Linear(dims[l], out_dim)

#             torch.nn.init.constant_(lin.bias, 0.0)
#             torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)

#             setattr(self, "lin" + str(l), lin)

#         self.softplus = nn.Softplus(beta=100)
#         self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
#         # NOTE cc custom ########
#         if deform_cc:
#             self.cc_linear = nn.Linear(128, 128)
#             self.cc = nn.Linear(128, 3)
#         #########################
#         self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
#         self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
#         torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
#         torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
#         if weight_norm:
#             self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
#         # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
#         torch.nn.init.constant_(self.blendshapes.bias, 0.0)
#         torch.nn.init.constant_(self.blendshapes.weight, 0.0)
#         torch.nn.init.constant_(self.skinning.bias, 0.0)
#         torch.nn.init.constant_(self.skinning.weight, 0.0)
#         # NOTE custom
#         if deform_cc:
#             torch.nn.init.constant_(self.cc_linear.bias, 0.0)
#             torch.nn.init.normal_(self.cc_linear.weight, 0.0, np.sqrt(2) / np.sqrt(128))
#             torch.nn.init.constant_(self.cc.bias, 0.0)
#             torch.nn.init.constant_(self.cc.weight, 0.0)

#         self.ghostbone = ghostbone

#     def canocanonical_deform(self, pnts_c, mask=None, cond=None):
#         assert self.deform_cc, 'deform_cc should be True'
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_scene_latent = cond['scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x = self.embed_fn(pnts_c)
#         else:
#             x = pnts_c
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_scene_latent = cond['scene_latent']
#             assert condition_scene_latent.shape[0] == x.shape[0], 'x dim: {}, condition_scene_latent dim: {}'.format(x.shape, condition_scene_latent.shape)
#             x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         pnts_c = pnts_c + self.cc(self.cc_linear(blendshapes[:, -131:-3]))     # NOTE offset을 더해준다. paper: canonical offset.
    
#         return pnts_c
    
#     def query_weights(self, pnts_c, mask=None, cond=None):
#         if mask is not None:
#             pnts_c = pnts_c[mask]
#             # custom #########################
#             if self.optimize_scene_latent_code:
#                 condition_scene_latent = cond['scene_latent'][mask]
#             ##################################

#         if self.embed_fn is not None:
#             x = self.embed_fn(pnts_c)
#         else:
#             x = pnts_c
        
#         # custom #########################
#         if self.optimize_scene_latent_code:
#             if mask is None:
#                 condition_scene_latent = cond['scene_latent']
#             assert condition_scene_latent.shape[0] == x.shape[0], 'double check'
#             x = torch.cat([x, condition_scene_latent], dim=1) # [300000, 71]
#         ##################################

#         for l in range(0, self.num_layers - 2):
#             lin = getattr(self, "lin" + str(l))
#             x = lin(x)
#             x = self.softplus(x)

#         blendshapes = self.blendshapes(x)
#         posedirs = blendshapes[:, :36 * 3]                                                                                          # NOTE 108 dim
#         shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]                                                               # NOTE 150 dim
#         beta_shapedirs = blendshapes[:, 36 * 3 + self.num_exp * 3:36 * 3 + self.num_exp * 3 + self.num_betashape * 3]               # NOTE 300 dim beta cancel
#         lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
#         # softmax implementation
#         lbs_weights_exp = torch.exp(20 * lbs_weights)
#         lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)
#         if self.deform_c:
#             pnts_c_flame = pnts_c + blendshapes[:, -3:]     # NOTE offset을 더해준다. paper: canonical offset.
#         else:
#             pnts_c_flame = pnts_c
#         return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame, beta_shapedirs.reshape(-1, 3, self.num_betashape)

#     def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None, cond=None, shapes=None):
#         shapedirs, posedirs, lbs_weights, pnts_c_flame, beta_shapedirs = self.query_weights(pnts_c, mask, cond)
#         pts_p = self.FLAMEServer.forward_pts(pnts_c=pnts_c_flame, 
#                                              betas=betas, 
#                                              transformations=transformations, 
#                                              pose_feature=pose_feature, 
#                                              shapedirs=shapedirs, 
#                                              posedirs=posedirs, 
#                                              lbs_weights=lbs_weights, 
#                                              beta_shapedirs=beta_shapedirs,
#                                              shapes=shapes,
#                                              dtype=torch.float32)
#         return pts_p, pnts_c_flame