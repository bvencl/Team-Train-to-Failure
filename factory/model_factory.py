import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

from factory.base_factory import BaseFactory


class ModelFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device=device)

        model = kwargs["config"].model.type
        transfer_learning = kwargs[
            "config"
        ].model.transfer_learning  # nem sok értelme van

        if model == "own":
            ...

        elif model == "mobilenet_v3_small":
            my_model = mobilenet_v3_small(
                weights=(
                    MobileNet_V3_Small_Weights.DEFAULT if transfer_learning else None
                )
            )

            my_model.features[0][0] = torch.nn.Conv2d(
                1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )

            my_model.classifier[3] = nn.Linear(
                my_model.classifier[3].in_features, kwargs["num_classes"]
            )
            
        elif model == "mobilenet_v3_large":
            my_model = mobilenet_v3_large(
                weights=(
                    MobileNet_V3_Large_Weights.DEFAULT if transfer_learning else None
                )
            )

            my_model.features[0][0] = torch.nn.Conv2d(
                1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )

            my_model.classifier[3] = nn.Linear(
                my_model.classifier[3].in_features, kwargs["num_classes"]
            )
            
        elif model == "efficientnet_v2_s":
            my_model = efficientnet_v2_s(
                weights=(
                    EfficientNet_V2_S_Weights.DEFAULT if transfer_learning else None
                )
            )

            # Modifying the first convolution layer for grayscale input
            my_model.features[0][0] = torch.nn.Conv2d(
                1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )

            # Modifying the classifier
            num_classes = kwargs.get("num_classes", 170)
            my_model.classifier[1] = nn.Linear(
                my_model.classifier[1].in_features, num_classes
            )
            
        elif model == "efficientnet_v2_m":
            my_model = efficientnet_v2_m(
                weights=(
                    EfficientNet_V2_M_Weights.DEFAULT if transfer_learning else None
                )
            )

            # Modifying the first convolution layer for grayscale input
            my_model.features[0][0] = torch.nn.Conv2d(
                1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )

            # Modifying the classifier
            num_classes = kwargs.get("num_classes", 170)
            my_model.classifier[1] = nn.Linear(
                my_model.classifier[1].in_features, num_classes
            )
            
        elif model == "efficientnet_v2_l":
            my_model = efficientnet_v2_l(
                weights=(
                    EfficientNet_V2_L_Weights.DEFAULT if transfer_learning else None
                )
            )

            # Modifying the first convolution layer for grayscale input
            my_model.features[0][0] = torch.nn.Conv2d(
                1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )

            # Modifying the classifier
            num_classes = kwargs.get("num_classes", 170)
            my_model.classifier[1] = nn.Linear(
                my_model.classifier[1].in_features, num_classes
            )
            
            
        else:
            raise NotImplementedError("Not a valid model option")

        return my_model.to(device)
