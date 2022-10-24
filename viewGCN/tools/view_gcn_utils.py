import torch
import torch.nn as nn
import torch.nn.functional as Functional

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint,stochastic=True):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if stochastic:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.zeros((B,),dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)            # 根据每个顶点之间的坐标算每个顶点之间的距离
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]      # topk的largest结果12列，每列4行表示离这个顶点最近的四个顶点索引torch.Size([20, 4, 12])
    id = torch.transpose(id, 1, 2)                  # torch.Size([20, 12, 4])
    return id

class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b,n,k,c = x.size()
        x = x.permute(0,3,1,2)

        y = self.avg_pool(x).view(b, c)         # torch.Size([20, 512])
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = x.permute(0,2,3,1)
        return x

class SELayer2(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, n = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        return x

class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.se = SELayer(channel=512)
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)          # k个邻居含自己的特征矩阵  torch.Size([20, 12, 4, 512])
        F = self.se(F)
        v = index_points(vertices,id)   # k个邻居含自己的坐标矩阵  torch.Size([20, 12, 4, 3])
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)         # 拿自己的坐标重复k遍  torch.Size([20, 12, 4, 3])
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)      # 自己坐标 邻居坐标 边坐标 边坐标的normal  torch.Size([20, 12, 4, 10])
        v_F = self.R(v_F)                                           # 经过一个非线性MLP  torch.Size([20, 12, 4, 10])变为torch.Size([20, 12, 4, 1])  将10变为1 相当于搞了一个权重
        F = torch.mul(v_F, F)                                       # 有点像算了一个注意力
        F = torch.sum(F,-2)                                         # 将k个节点的特征加起来
        return F

class KNN_dist_raw(nn.Module):
    def __init__(self,k):
        super(KNN_dist_raw, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F,-2)
        return F

class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view, nclasses):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.nclasses = nclasses
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, self.nclasses*self.s_views))
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views,stochastic=self.training)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(
            F.shape[0], k, self.s_views, self.nclasses).transpose(1, 2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            SELayer2(512),
            nn.LeakyReLU(0.2),
        )

        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k).cuda()
    def forward(self,F,V):
        F = self.KNN(F, V)          # 算了K近邻后的特征
        F = F.permute(0,2,1)
        F = self.conv(F)
        F = F.permute(0, 2, 1)
        return F

class LocalGCN_raw(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN_raw,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist_raw(k=self.k).cuda()
    def forward(self,F,V):
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F
