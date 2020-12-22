from torchvision.models.resnet import conv3x3, conv1x1
import torch
import torch.nn as nn

class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes: int = 1000,
        zero_init_residual: bool = True,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        dropout_prob: float = 0.3,
        stochastic_depth_prob: float = 0.5
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.dropout_prob = dropout_prob
        self.prob_delta = (1 - stochastic_depth_prob) / ((sum(layers) - 1))
        self.layer_num = 0
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        
        sd_prob = 1.0 - self.prob_delta * self.layer_num
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            self.dropout_prob, sd_prob))
        self.layer_num += 1
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            sd_prob = 1.0 - self.prob_delta * self.layer_num
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                dropout_prob=self.dropout_prob,
                                stochastic_depth_prob=sd_prob))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 dropout_prob=0.0, stochastic_depth_prob=1.0):
        assert 0.0 <= stochastic_depth_prob <= 1.0
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        if dropout_prob == 0.0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout_prob)
        
        sd_prob = torch.tensor([stochastic_depth_prob], dtype=torch.float, requires_grad=False)
        self.register_buffer("stochastic_depth_prob", sd_prob)

    def f(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        one = torch.ones(1, device=self.stochastic_depth_prob.device, requires_grad=False)
        
        # not using stochastic depth
        if torch.equal(self.stochastic_depth_prob, one):
            out = identity + self.f(x)

        # using stochastic depth at training
        elif self.training:
            if torch.equal(torch.bernoulli(self.stochastic_depth_prob), one):
                out = identity + self.f(x)
            else:
                return identity

        # using stochastic depth at inference
        else:
            out = identity + self.f(x) * self.stochastic_depth_prob

        return self.relu(out)
    

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet26(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3, 3], **kwargs)

def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def load_model(model_name, **kwargs):
    return globals()[model_name](**kwargs)  
        
    
def save_model(model, optimizer, scheduler, epoch, phase, best_acc, best_epoch, best_model, save_path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_model": best_model
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    print(locals()["resnet18"])