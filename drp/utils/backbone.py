import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from drp.utils.ConvLSTM import ConvLSTM


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 feature_dim=512,
                 activation="relu",
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.activation = activation

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=conv1_t_size,
                               padding=conv1_t_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.nb_feature = block_inplanes[3] * block.expansion

        self.fc = nn.Linear(self.nb_feature, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.activation == "relu":
            x = self.relu(x)

        elif self.activation == "prelu":
            x = self.prelu(x)

        else:
            raise EOFError

        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetModified(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass arguments to the parent ResNet class

    def forward(self, x):
        # Initial convolution, normalization, and activation
        x = self.conv1(x)
        x = self.bn1(x)
        if self.activation == "relu":
            x = self.relu(x)
        elif self.activation == "prelu":
            x = self.prelu(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation}")

        # Optional max pooling
        if not self.no_max_pool:
            x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layer (output prediction)
        x_ = self.fc(x)

        # Return feature embeddings and output predictions
        return x, x_


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


def generate_model_modified(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNetModified(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNetModified(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNetModified(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNetModified(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNetModified(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNetModified(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNetModified(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class Simpler3DNet(nn.Module):
    """
    Model from the paper https://link.springer.com/content/pdf/10.1007/s10596-020-09941-w.pdf

    This model suffers from the problems of cuda/cudnn instability with respect to tensor shapes

    Currently impossible to have large batch size filling V100 32 Go. See PyTorch issues :
    - https://github.com/pytorch/pytorch/issues/52211
    - https://github.com/pytorch/pytorch/issues/51776

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv3d-1       [-1, 32, 96, 96, 96]           4,032
           BatchNorm3d-2       [-1, 32, 19, 19, 19]              64
                Conv3d-3       [-1, 64, 15, 15, 15]         256,064
           BatchNorm3d-4          [-1, 64, 3, 3, 3]             128
                Conv3d-5         [-1, 128, 1, 1, 1]         221,312
           BatchNorm3d-6         [-1, 128, 1, 1, 1]             256
                Linear-7                   [-1, 64]           8,256
                Linear-8                    [-1, 1]              65
    ================================================================
    Total params: 490,177
    Trainable params: 490,177
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 3.81
    Forward/backward pass size (MB): 219.34
    Params size (MB): 1.87
    Estimated Total Size (MB): 225.02
    ----------------------------------------------------------------
    """

    def __init__(self, activation="prelu", dilation=1):
        super(Simpler3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, padding=0, dilation=dilation)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, padding=0, dilation=dilation)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=0, dilation=dilation)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        if activation == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.act4 = nn.ReLU()
        elif activation == "prelu":
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
            self.act3 = nn.PReLU()
            self.act4 = nn.PReLU()
        else:
            activations = ["relu", "prelu"]
            raise ValueError(
                f"undefined activation (get: '{activation}' expected: {activations}"
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        print("x.shape", x.shape)
        x = F.max_pool3d(x, 5)
        x = self.bn1(x)
        print("x.shape", x.shape)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = F.max_pool3d(x, 5)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)

        x = x.view(-1, 128)

        x = self.act4(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Simpler3DNet_dilated(nn.Module):
    """
    Model from the paper https://link.springer.com/content/pdf/10.1007/s10596-020-09941-w.pdf

    This model suffers from the problems of cuda/cudnn instability with respect to tensor shapes

    Currently impossible to have large batch size filling V100 32 Go. See PyTorch issues :
    - https://github.com/pytorch/pytorch/issues/52211
    - https://github.com/pytorch/pytorch/issues/51776

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv3d-1       [-1, 32, 96, 96, 96]           4,032
           BatchNorm3d-2       [-1, 32, 19, 19, 19]              64
                Conv3d-3       [-1, 64, 15, 15, 15]         256,064
           BatchNorm3d-4          [-1, 64, 3, 3, 3]             128
                Conv3d-5         [-1, 128, 1, 1, 1]         221,312
           BatchNorm3d-6         [-1, 128, 1, 1, 1]             256
                Linear-7                   [-1, 64]           8,256
                Linear-8                    [-1, 1]              65
    ================================================================
    Total params: 490,177
    Trainable params: 490,177
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 3.81
    Forward/backward pass size (MB): 219.34
    Params size (MB): 1.87
    Estimated Total Size (MB): 225.02
    ----------------------------------------------------------------
    """

    def __init__(self, activation="prelu", dilation=2):
        super(Simpler3DNet_dilated, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, padding=2, dilation=dilation)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, padding=2, dilation=dilation)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)

        if activation == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
            self.act3 = nn.ReLU()
            self.act4 = nn.ReLU()
        elif activation == "prelu":
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
            self.act3 = nn.PReLU()
            self.act4 = nn.PReLU()
        else:
            activations = ["relu", "prelu"]
            raise ValueError(
                f"undefined activation (got: '{activation}', expected: {activations})"
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = F.max_pool3d(x, 5)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool3d(x, 5)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.bn3(x)

        # Ensure the tensor is flattened correctly before feeding into the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.act4(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Custom_ConvLSTM(nn.Module):
    def __init__(self, hidden_dim=[16, 32, 64, 128]):
        super(Custom_ConvLSTM, self).__init__()

        self.ConvLSTM1 = ConvLSTM(input_dim=1, hidden_dim=hidden_dim[0], kernel_size=(3, 3), num_layers=1, batch_first=True)
        self.BatchNorm1 = nn.BatchNorm3d(100)
        self.Pooling1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.ConvLSTM2 = ConvLSTM(input_dim=hidden_dim[0], hidden_dim=hidden_dim[1], kernel_size=(3, 3), num_layers=1, batch_first=False)
        self.BatchNorm2 = nn.BatchNorm3d(100)
        self.Pooling2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.ConvLSTM3 = ConvLSTM(input_dim=hidden_dim[1], hidden_dim=hidden_dim[2], kernel_size=(3, 3), num_layers=1, batch_first=False)
        self.BatchNorm3 = nn.BatchNorm3d(100)
        self.Pooling3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.ConvLSTM4 = ConvLSTM(input_dim=hidden_dim[2], hidden_dim=hidden_dim[3], kernel_size=(3, 3), num_layers=1, batch_first=False)
        self.BatchNorm4 = nn.BatchNorm3d(100)
        self.Pooling4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fc1 = nn.Linear(hidden_dim[3]*100*8*8, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.ConvLSTM1(x)
        x = self.BatchNorm1(x)
        x = self.Pooling1(x)

        x = self.ConvLSTM2(x)
        x = self.BatchNorm2(x)
        x = self.Pooling2(x)

        x = self.ConvLSTM3(x)
        x = self.BatchNorm3(x)
        x = self.Pooling3(x)

        x = self.ConvLSTM4(x)
        x = self.BatchNorm4(x)
        x = self.Pooling4(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    model = generate_model(model_depth=18)

    # Create a dummy input with the specified size (batch_size, channels, depth, height, width)
    dummy_input = torch.randn(1, 1, 100, 100, 100)

    # Perform a forward pass with the dummy input
    output = model(dummy_input)