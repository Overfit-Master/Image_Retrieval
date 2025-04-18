import warnings
import torchvision
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)
flower_model = torchvision.models.resnet18(pretrained=True)
car_model = torchvision.models.resnet18(pretrained=True)
classify_model = torchvision.models.resnet18(pretrained=True)
warnings.resetwarnings()

flower_classes = 17
car_classes = 196
classify_classes = 2

flower_model.fc = nn.Linear(flower_model.fc.in_features, flower_classes)
car_model.fc = nn.Linear(car_model.fc.in_features, car_classes)
classify_model.fc = nn.Linear(classify_model.fc.in_features, classify_classes)