import torch
import torch.nn as nn


class MVModel(nn.Module):
    def __init__(self, num_views, num_class, backbone, feat_size):

        super().__init__()
        self.num_views = num_views
        self.num_class = num_class
        self.dropout_p = 0.5
        self.feat_size = feat_size

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = nn.Sequential(*img_layers)

        self.final_fc = MVFC(
            num_views=self.num_views,
            in_features=in_features,
            out_features=self.num_class,
            dropout_p=self.dropout_p)

    def forward(self, img):
        """
        :param img: (B*N)*C*W*H
        :return:
        """
        img = torch.flatten(img, start_dim=0, end_dim=1)
        feat = self.img_model(img)
        logit = self.final_fc(feat)
        return logit,None

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(3, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features


class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
                BatchNormPoint(in_features),
                # dropout before concatenation so that each view drops features independently
                nn.Dropout(dropout_p),
                nn.Flatten(),
                nn.Linear(in_features=in_features * self.num_views,
                          out_features=in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(in_features=in_features, out_features=out_features,
                          bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        out = self.model(feat)
        return out


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return inp.squeeze()

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)