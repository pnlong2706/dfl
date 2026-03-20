import torch
from torch import nn

from nebula.core.models.nebulamodel import NebulaModel


class CIFAR100ModelSimpleNet(NebulaModel):
    """
    SimpleNet architecture for CIFAR-100 (simplenet_cifar_5m variant).

    Based on: "Lets keep it simple, Using simple architectures to outperform
              deeper and more complex architectures" (Hasanpour et al., 2016)
    Paper: https://arxiv.org/abs/1608.06037

    Achieves strong accuracy on CIFAR-100.

    Architecture: Conv(3x3) + BatchNorm + ReLU + MaxPool
    - Uses BatchNorm (compatible with pseudo aggregation via buffer skipping)
    - Light dropout at pool layers
    - 1x1 convolutions for dimensionality control
    - ~5M parameters
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # SimpleNet CIFAR-100 architecture (simplenet_cifar_5m)
        # Same as CIFAR-10 but with 100 output classes
        # Format: Conv -> BN -> ReLU

        self.features = nn.Sequential(
            # Block 1: 32x32
            self._make_conv_bn_relu(input_channels, 64),
            self._make_conv_bn_relu(64, 128),
            self._make_conv_bn_relu(128, 128),
            self._make_conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            # Block 2: 16x16
            self._make_conv_bn_relu(128, 128),
            self._make_conv_bn_relu(128, 128),
            self._make_conv_bn_relu(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8

            # Block 3: 8x8
            self._make_conv_bn_relu(256, 256),
            self._make_conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4

            # Block 4: 4x4 (1x1 convolutions)
            self._make_conv_bn_relu(256, 512, kernel_size=3),
            self._make_conv_bn_relu(512, 2048, kernel_size=1),
            self._make_conv_bn_relu(2048, 256, kernel_size=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2

            # Final conv
            self._make_conv_bn_relu(256, 256),
        )

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_conv_bn_relu(self, in_channels, out_channels, kernel_size=3):
        """Create Conv -> BatchNorm -> ReLU block"""
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.05),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        """Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        # SGD with momentum (as used in the paper)
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        return optimizer
