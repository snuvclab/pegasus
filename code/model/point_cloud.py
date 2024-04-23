import torch
import torch.nn as nn
import numpy as np
import time
import os

class PointCloudPEGASUS(nn.Module):
    '''
    NOTE
    1) register_parameter를 수정할때마다 해줬음. 기존에는 self.points = nn.Parameter(...) 이런식으로 해줬었음.
    2) init을 맨처음에는 다 해주는 방식으로 바꿨음. self.init은 다른 모듈에서 불러올때나 쓰겠금.
    '''
    def __init__(
        self,
        n_init_points,
        max_points=131072,
        min_radius=0.0053,
        init_radius=0.5,
        radius_factor=0.3
    ):
        super(PointCloudPEGASUS, self).__init__()
        self.radius_factor = radius_factor
        self.max_points = max_points
        self.init_radius = init_radius
        self.min_radius = min_radius
        
        self.init(n_init_points)
        #############################

    def init(self, n_init_points):
        # NOTE train하고 test에서 호출됨.
        print("[INFO] current point number: ", n_init_points)
        # initialize sphere
        init_points = torch.rand(n_init_points, 3) * 2.0 - 1.0
        init_normals = nn.functional.normalize(init_points, dim=1)
        init_points = init_normals * self.init_radius
        self.register_parameter("points", nn.Parameter(init_points))                                                        # NOTE self.pc.points. Make trainable by default. require gradient computation

    def prune(self, visible_points):
        """Prune not rendered points"""
        self.points = nn.Parameter(self.points.data[visible_points])                                                      # NOTE original code
        print(
            "[INFO] Pruning points, original: {}, new: {}".format(
                len(visible_points), sum(visible_points)
            )
        )

    def upsample_points(self, new_points):
        self.points = nn.Parameter(torch.cat([self.points.to(new_points.device), new_points], dim=0))                                        # NOTE original code
    
    def points_parameter(self, points):
        self.points = nn.Parameter(points)