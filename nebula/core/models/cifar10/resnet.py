import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from torchmetrics import MetricCollection

from nebula.core.models.nebulamodel import NebulaModel

matplotlib.use("Agg")
plt.switch_backend("Agg")
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 32

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

def get_resnet_model(classifier_name, num_classes=10, input_channels=3):
    """
    Get a ResNet model adapted for CIFAR-10/100.

    CIFAR images are 32x32, much smaller than ImageNet's 224x224,
    so we need to adapt the architecture:
    - Use 3x3 conv with stride 1 instead of 7x7 conv with stride 2
    - Remove the initial max pooling layer
    """
    if classifier_name == "resnet18":
        model = resnet18(weights=None)
    elif classifier_name == "resnet34":
        model = resnet34(weights=None)
    elif classifier_name == "resnet50":
        model = resnet50(weights=None)
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Adapt the first convolutional layer for CIFAR (32x32 images)
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the max pooling layer (images are already small)
    model.maxpool = nn.Identity()

    # Replace the final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class CIFAR10ModelResNet(NebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        implementation="scratch",
        classifier="resnet9",
    ):
        # Create metrics before calling super().__init__()
        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=num_classes),
                MulticlassPrecision(num_classes=num_classes),
                MulticlassRecall(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ])

        # Don't create confusion_matrix here - let parent handle it
        # Parent will create both cm and cm_global when confusion_matrix=None

        # Pass all parameters to parent class
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,  # Pass as-is (usually None)
            seed=seed
        )

        self.implementation = implementation
        self.classifier = classifier

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = self._build_model(input_channels, num_classes)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def _build_model(self, input_channels, num_classes):
        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                """
                ResNet9 implementation
                """

                def conv_block(input_channels, num_classes, pool=False):
                    layers = [
                        nn.Conv2d(input_channels, num_classes, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_classes),
                        nn.ReLU(inplace=True),
                    ]
                    if pool:
                        layers.append(nn.MaxPool2d(2))
                    return nn.Sequential(*layers)

                conv1 = conv_block(input_channels, 64)
                conv2 = conv_block(64, 128, pool=True)
                res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

                conv3 = conv_block(128, 256, pool=True)
                conv4 = conv_block(256, 512, pool=True)
                res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

                classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))

                return nn.ModuleDict({
                    "conv1": conv1,
                    "conv2": conv2,
                    "res1": res1,
                    "conv3": conv3,
                    "conv4": conv4,
                    "res2": res2,
                    "classifier": classifier,
                })

            elif self.classifier in ["resnet18", "resnet34", "resnet50"]:
                # Use torchvision ResNet models adapted for CIFAR
                return get_resnet_model(self.classifier, num_classes, input_channels)

            else:
                raise NotImplementedError(f"Classifier {self.classifier} not implemented for scratch mode")

        elif self.implementation == "timm":
            raise NotImplementedError("TIMM implementation not yet supported")

        else:
            raise NotImplementedError(f"Implementation {self.implementation} not supported")

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(x)}")

        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                out = self.model["conv1"](x)
                out = self.model["conv2"](out)
                out = self.model["res1"](out) + out
                out = self.model["conv3"](out)
                out = self.model["conv4"](out)
                out = self.model["res2"](out) + out
                out = self.model["classifier"](out)
                return out
            elif self.classifier in ["resnet18", "resnet34", "resnet50"]:
                # For torchvision ResNet models
                return self.model(x)
            else:
                raise NotImplementedError(f"Forward not implemented for classifier {self.classifier}")

        elif self.implementation == "timm":
            raise NotImplementedError("TIMM implementation not yet supported")

        else:
            raise NotImplementedError(f"Implementation {self.implementation} not supported")

    def configure_optimizers(self):
        if self.implementation == "scratch" and self.classifier == "resnet9":
            # ResNet9 uses ModuleDict, need to collect params from all modules
            params = []
            for key, module in self.model.items():
                params += list(module.parameters())
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        else:
            # For ResNet-18, ResNet-34, ResNet-50 and other implementations
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer
