import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
import numpy as np
import PIL.Image as Image
import ast


weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\flower-2024-03-31-16-55_best.pt"
warnings.simplefilter("ignore")
model = models.resnet18(pretrained=True)
warnings.resetwarnings()
model.fc = nn.Linear(model.fc.in_features, 17)
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)
# model.to("cuda:0")
feature_model = copy.deepcopy(model)
feature_model.fc = nn.Identity()
# model.eval()
print(model)
model.load_state_dict(torch.load(weight_path))


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# print(model)
# print(feature_model)

image_path_1 = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data\database\image_0007_Daffodil.jpg"
image_path_2 = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data\retrival\image_0001_Daffodil.jpg"
image_path_3 = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data\retrival\image_0083_Snowdrop.jpg"
image_1 = Image.open(image_path_1).convert("RGB")
tensor_1 = transform(image_1).unsqueeze(0)      # 升维操作，加入维度batch_size

image_2 = Image.open(image_path_2).convert("RGB")
tensor_2 = transform(image_2).unsqueeze(0)

image_3 = Image.open(image_path_3).convert("RGB")
tensor_3 = transform(image_3).unsqueeze(0)
with torch.no_grad():
    output1 = feature_model(tensor_1.to("cuda:0"))
    # output1 = output1.view(output1.size(0), -1)
    output2 = feature_model(tensor_2.to("cuda:0"))
    # output2 = output2.view(output2.size(0), -1)
    output3 = feature_model(tensor_3.to("cuda:0"))


print(output2.shape)
print(type(output1))
a = np.array2string(output1.cpu().detach().numpy())
b = torch.from_numpy()

cos1 = F.cosine_similarity(output1, output3)
print(cos1)