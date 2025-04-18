import os
import random
from abc import ABC

import torch
import torchvision.transforms as transform
import PIL.Image as Image
from torch.utils.data import Dataset as Dataset

flower_root_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data"
car_root_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Stanford_Cars\Divide_data"

# 训练数据采用论文的处理方式，数据库和检索数据采用普通处理
train_transform = transform.Compose([
    # 将图像尺寸变为250*250
    transform.Resize([250, 250]),
    # 将图像进行概率为0.5的水平翻转
    transform.RandomHorizontalFlip(p=0.5),
    # 将图像随机裁剪至224*224
    transform.RandomCrop([224, 224]),
    # 将图像转为张量格式
    transform.ToTensor(),
    # 对图像数据进行标准化
    transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transform.Compose([
    transform.Resize([224, 224]),
    transform.ToTensor()
]
)


# 通过所给信息获取到对应的txt路径，并返回相应的图像路径和类别列表
def get_path_list(task: str, use: str):
    """
    :param task: "flower" or "car"
    :param use: "train", "Database" or "retrival"
    :return: image path list and label list
    """
    if task == "flower":
        txt_root_path = flower_root_path
    else:
        txt_root_path = car_root_path
    txt_file_name = use + '.txt'
    txt_path = os.path.join(txt_root_path, txt_file_name)

    path_list = []
    label_list = []
    with open(txt_path, "r") as f:
        for msg in f.readlines():
            path_list.append(msg.split(',')[0])
            label_list.append(msg.split(',')[1][:-1])
    return path_list, label_list


class My_Dataset(Dataset, ABC):
    def __init__(self, task: str, use: str):
        if use == "train" or use == "all":
            self.transform = train_transform
        else:
            self.transform = eval_transform

        path_list, label_list = get_path_list(task, use)
        self.path_list = path_list
        self.label_list = label_list

    def __getitem__(self, index):
        image = self.path_list[index]
        label = int(self.label_list[index])

        image = Image.open(image).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(label).long()

        return image, label, index

    def __len__(self):
        return len(self.path_list)


if __name__ == '__main__':
    a, b = get_path_list("car", "train")
    for i, j in zip(a, b):
        print(i, j)