import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_small

from factory.base_factory import BaseFactory


class ModelFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device=device)

        model = kwargs["config"].model.type
        transfer_learning = kwargs["config"].model.transfer_learning # nem sok értelme van


        if model == "own":
            ...

        elif model == "mobilenet_v3_small":
            my_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if transfer_learning else None)
            #! setting first kernel's size?
            my_model.classifier[3] = nn.Linear(my_model.classifier[3].in_features, kwargs["num_classes"])
        else:
            raise NotImplementedError("Not a valid model option")

        return my_model
