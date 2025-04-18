import warnings
import torchvision
import torch.nn as nn

# pretrained的用法是老版本，所以会报警告，此处忽略处理，可以使用weight进行加载
warnings.simplefilter("ignore", UserWarning)
flower_model = torchvision.models.resnet18(pretrained=True)
car_model = torchvision.models.resnet18(pretrained=True)
classify_model = torchvision.models.resnet18(pretrained=True)
warnings.resetwarnings()

# 花的类别为17，车的类别为196， 2为车花二分类模型的类别数
flower_classes = 17
car_classes = 196
classify_classes = 2

flower_model.fc = nn.Linear(flower_model.fc.in_features, flower_classes)
car_model.fc = nn.Linear(car_model.fc.in_features, car_classes)
classify_model.fc = nn.Linear(classify_model.fc.in_features, classify_classes)
