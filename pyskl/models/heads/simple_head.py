import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes, bias= False)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        # normal_init(self.fc_cls, std=self.init_std)
        nn.init.kaiming_normal_(self.fc_cls.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score


@HEADS.register_module()
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)
    # def forward(self, x):
    #     # ... existing code ...
    #     cls_score = self.fc_cls(x)
    #     # If you have average_clips:
    #     if self.average_clips:
    #         cls_score = cls_score.mean(dim=0, keepdim=True)
    #     return cls_score


@HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HEADS.register_module()
class I3DHeadWithDemographics(I3DHead):
    """I3DHead extended with a demographics fusion branch.

    Backbone features and demographics embeddings are concatenated before the
    final FC, so neither branch can dominate independently:
        feat  = pool(backbone)               # (N, in_channels)
        emb   = clamp(relu(demo_fc1(demo)))  # (N, demo_hidden_channels)
        output = fusion_fc(cat([feat, emb])) # (N, num_classes)

    Args:
        demo_in_channels (int): Number of demographic features (default 5).
        demo_hidden_channels (int): Width of the demographics embedding. Default: 32.
        demo_clamp (float): Clamp bound applied to the demo embedding activations.
            Limits demographic influence on the final prediction. Default: 1.0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='MSELoss', loss_weight=1.0),
                 dropout=0.5,
                 init_std=0.01,
                 demo_in_channels=5,
                 demo_hidden_channels=32,
                 demo_clamp=1.0,
                 **kwargs):
        super().__init__(num_classes, in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         **kwargs)
        self.demo_clamp = demo_clamp
        self.demo_fc1 = nn.Linear(demo_in_channels, demo_hidden_channels, bias=True)
        self.demo_relu = nn.ReLU(inplace=True)
        # Fusion layer replaces fc_cls for the demographics path
        self.fusion_fc = nn.Linear(in_channels + demo_hidden_channels, num_classes, bias=True)

    def init_weights(self):
        super().init_weights()
        nn.init.kaiming_normal_(self.demo_fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.demo_fc1.bias)
        nn.init.normal_(self.fusion_fc.weight, std=0.01)
        nn.init.zeros_(self.fusion_fc.bias)

    def forward(self, x, demographics=None):
        """
        Args:
            x: backbone feature tensor (N, C, T, H, W) or (N, C).
            demographics: float tensor of shape (N, demo_in_channels).
        """
        if len(x.shape) != 2:
            pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(x, (tuple, list)):
                x = torch.cat(x, dim=1)
            x = pool(x)
            x = x.view(x.shape[:2])

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        if demographics is None:
            return self.fc_cls(x)

        # Clamp embedding to bound demographic influence, then fuse
        demo_emb = self.demo_relu(self.demo_fc1(demographics)).clamp(-self.demo_clamp, self.demo_clamp)
        return self.fusion_fc(torch.cat([x, demo_emb], dim=1))


@HEADS.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


@HEADS.register_module()
class GCNHeadWithDemographics(GCNHead):
    """GCNHead extended with a demographics fusion branch (same design as I3DHeadWithDemographics)."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='MSELoss', loss_weight=1.0),
                 dropout=0.,
                 init_std=0.01,
                 demo_in_channels=5,
                 demo_hidden_channels=32,
                 demo_clamp=1.0,
                 **kwargs):
        super().__init__(num_classes, in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         **kwargs)
        self.demo_clamp = demo_clamp
        self.demo_fc1 = nn.Linear(demo_in_channels, demo_hidden_channels, bias=True)
        self.demo_relu = nn.ReLU(inplace=True)
        self.fusion_fc = nn.Linear(in_channels + demo_hidden_channels, num_classes, bias=True)

    def init_weights(self):
        super().init_weights()
        nn.init.kaiming_normal_(self.demo_fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.demo_fc1.bias)
        nn.init.normal_(self.fusion_fc.weight, std=0.01)
        nn.init.zeros_(self.fusion_fc.bias)

    def forward(self, x, demographics=None):
        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            pool = nn.AdaptiveAvgPool2d(1)
            N, M, C, T, V = x.shape
            x = x.reshape(N * M, C, T, V)
            x = pool(x)
            x = x.reshape(N, M, C)
            x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        if demographics is None:
            return self.fc_cls(x)

        demo_emb = self.demo_relu(self.demo_fc1(demographics)).clamp(-self.demo_clamp, self.demo_clamp)
        return self.fusion_fc(torch.cat([x, demo_emb], dim=1))


@HEADS.register_module()
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)
