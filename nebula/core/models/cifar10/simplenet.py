import torch
from torch import nn

from nebula.core.models.nebulamodel import NebulaModel


class CIFAR10ModelSimpleNet(NebulaModel):
    """
    SimpleNet architecture for CIFAR-10.

    Based on: "Lets keep it simple, Using simple architectures to outperform
              deeper and more complex architectures"
    Paper: https://arxiv.org/abs/1608.06037

    Key features:
    - No BatchNorm (works well with federated learning and pseudo aggregation)
    - Heavy use of dropout for regularization
    - 1x1 convolutions for dimensionality reduction
    - Achieves 95.32% on CIFAR-10 (better than ResNet-18!)

    Architecture: Pure Conv + ReLU + Dropout + MaxPool
    Parameters: ~5.5M (similar to ResNet-18's 11M)
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        dropout_rate=0.2,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.dropout_rate = dropout_rate
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # Build the network following the SimpleNet architecture
        self.features = nn.Sequential(
            # Block 1: 32x32x3 -> 32x32x64
            nn.Dropout(dropout_rate),
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 2: 32x32x64 -> 16x16x128
            nn.Dropout(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x16x128 -> 16x16x128
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 4: 16x16x128 -> 8x8x128
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5: 8x8x128 -> 8x8x256
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Block 6: 8x8x256 -> 4x4x256
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 7: 4x4x256 -> 2x2x512
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 8: 2x2x512 -> 2x2x2048 (1x1 conv for channel expansion)
            nn.Dropout(dropout_rate),
            nn.Conv2d(512, 2048, kernel_size=1),
            nn.ReLU(inplace=True),

            # Block 9: 2x2x2048 -> 2x2x256 (1x1 conv for dimensionality reduction)
            nn.Dropout(dropout_rate),
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),

            # Block 10: 2x2x256 -> 2x2x num_classes (1x1 conv for classification)
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, num_classes, kernel_size=1),

            # Global average pooling: 2x2x num_classes -> 1x1x num_classes
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def configure_optimizers(self):
        # Following the paper's training setup
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True
        )
        return optimizer
