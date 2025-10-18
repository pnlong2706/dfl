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
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 32

BATCH_SIZE = 256 if torch.cuda.is_available() else 64


def get_resnet_model(classifier_name, num_classes=100, input_channels=3):
    """
    Get a ResNet model adapted for CIFAR-100.

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

    # Replace the final fully connected layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class CIFAR100ModelResNet(NebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        implementation="scratch",
        classifier="resnet18",
    ):
        # Create metrics before calling super().__init__()
        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=num_classes),
                MulticlassPrecision(num_classes=num_classes),
                MulticlassRecall(num_classes=num_classes),
                MulticlassF1Score(num_classes=num_classes),
            ])

        if confusion_matrix is None:
            confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # Pass all parameters to parent class
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed
        )

        self.implementation = implementation
        self.classifier = classifier

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = self._build_model(input_channels, num_classes)

        # Parent class already creates epoch_global_number, but with different keys
        # Keep this for backward compatibility
        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def _build_model(self, input_channels, num_classes):
        if self.implementation == "scratch":
            if self.classifier in ["resnet18", "resnet34", "resnet50"]:
                # Use torchvision ResNet models adapted for CIFAR-100
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
            if self.classifier in ["resnet18", "resnet34", "resnet50"]:
                # For torchvision ResNet models
                return self.model(x)
            else:
                raise NotImplementedError(f"Forward not implemented for classifier {self.classifier}")

        elif self.implementation == "timm":
            raise NotImplementedError("TIMM implementation not yet supported")

        else:
            raise NotImplementedError(f"Implementation {self.implementation} not supported")

    def configure_optimizers(self):
        # For all ResNet variants
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer
