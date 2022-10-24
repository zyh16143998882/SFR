# to import files from parent dir
import sys
import os

from models.blocks import MLP, Conv1dLayer
from models.pointnet import PointNet2, load_point_ckpt, load_point_ckpt2, PointNet3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ops import mvctosvc
from util import batch_tensor, unbatch_tensor
import torch
import numpy as np
from torch import nn
from torch._six import inf
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



# class ViewMaxAgregate(nn.Module):
#     def __init__(self,  model):
#         super().__init__()
#         self.model = model
#
#     def forward(self, mvimages):
#         B, M, C, H, W = mvimages.shape
#         pooled_view = torch.max(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True))[0], B, dim=1, unsqueeze=True), dim=1)[0]
#         return pooled_view.squeeze()

class ViewMaxAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.max(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)[0]
        return pooled_view.squeeze()

class ViewAvgAgregate(nn.Module):
    def __init__(self,  model):
        super().__init__()
        self.model = model

    def forward(self, mvimages):
        B, M, C, H, W = mvimages.shape
        pooled_view = torch.mean(unbatch_tensor(self.model(batch_tensor(
            mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True), dim=1)
        return pooled_view.squeeze()


class UnrolledDictModel(nn.Module):
    "a helper class that unroll pytorch models that return dictionaries instead of tensors"

    def __init__(self,  model, keyword="out"):
        super().__init__()
        self.model = model
        self.keyword = keyword

    def forward(self, x):
        return self.model(x)[self.keyword]

class MVAgregate(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes)
        )
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)

    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view

# class MVAgregate2(nn.Module):
#     def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
#         super().__init__()
#         self.agr_type = agr_type
#         # self.fc = nn.Sequential(
#         #     nn.LayerNorm(feat_dim*2),
#         #     nn.Linear(feat_dim * 2, feat_dim),
#         #     nn.LayerNorm(feat_dim),
#         #     nn.Linear(feat_dim, num_classes),
#         # )
#         self.fc = nn.Sequential(
#             nn.LayerNorm(feat_dim),
#             nn.Linear(feat_dim, num_classes)
#         )
#         # self.fc = MLP([512, num_classes],
#         #                        act='relu', norm=True, bias=True, dropout=0.5)
#         if self.agr_type == "max":
#             self.aggregation_model = ViewMaxAgregate(model=model)
#         elif self.agr_type == "mean":
#             self.aggregation_model = ViewAvgAgregate(model=model)
#
#         self.pointnet = PointNet2(40, alignment=True)
#         load_point_ckpt2(self.pointnet, "PointNet",ckpt_dir='./checkpoint')
#
#
#
#
#     def forward(self, mvimages, points):
#         pooled_view = self.aggregation_model(mvimages)
#         feat3d = self.pointnet(points)[0]
#         mean = torch.mean(pooled_view,dim=1).unsqueeze(1)
#         var = torch.var(pooled_view,dim=1).unsqueeze(1)
#         feat3d = feat3d*var + mean
#         pooled_view = torch.cat([pooled_view.unsqueeze(1),feat3d.unsqueeze(1)],dim=1)
#         pooled_view = torch.max(pooled_view, dim=1, keepdim=True)[0].squeeze()
#
#
#         predictions = self.fc(pooled_view)
#         return predictions, pooled_view

class MVAgregate2(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type

        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(feat_dim, num_classes)
        )
        # self.fc = MLP([1024, 512, 256, num_classes],
        #                       act='relu', norm=True, bias=True, dropout=0.5)
        self.conv1 = Conv1dLayer(
            [512, 512, 1024], act='relu', norm=True, bias=True)
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)

        self.pointnet = PointNet3(40, alignment=True)
        # load_point_ckpt2(self.pointnet, "PointNet",ckpt_dir='./checkpoint')




    def forward(self, mvimages, points):
        pooled_view = self.aggregation_model(mvimages)
        feat3d = self.pointnet(points)[1]
        mean = torch.mean(pooled_view,dim=1).unsqueeze(1)
        var = torch.var(pooled_view,dim=1).unsqueeze(1)
        feat3d = feat3d*var + mean
        x = torch.cat([pooled_view.unsqueeze(2),feat3d.unsqueeze(2)],dim=2)
        x = self.conv1(x)
        # x = torch.max(x, dim=2, keepdim=True)[0]
        x = torch.mean(x, dim=2, keepdim=True)
        global_feature = x.view(-1, 1024)



        predictions = self.fc(global_feature)
        return predictions, pooled_view

class MVAgregate3(nn.Module):
    def __init__(self,  model, agr_type="max", feat_dim=512, num_classes=1000):
        super().__init__()
        self.agr_type = agr_type
        self.fc = MLP([1024, 512, 256, num_classes],
                              act='relu', norm=True, bias=True, dropout=0.5)
        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAgregate(model=model)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAgregate(model=model)

    def forward(self, mvimages):
        pooled_view = self.aggregation_model(mvimages)
        predictions = self.fc(pooled_view)
        return predictions, pooled_view


