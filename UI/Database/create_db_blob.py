import os
import sqlite3
import torch.nn as nn
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import io
from tqdm import tqdm
import copy

import Retrival.model as Model

# 数据库路径
db_path = "./retrieval.db"

# 模型文件路径
flower_weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\flower-2024-03-31-16-55_best.pt"
car_weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\car-2024-04-01-21-13_best.pt"

# 数据集存放路径
flower_database_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data\database"
car_database_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Stanford_Cars\Divide_data\database"
# 提取图像特征的数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                         )
]
)


# 创建数据库
def create_table(db, tb, dir_path):
    # 根据所创造的表加载相应模型
    if tb == "flower":
        model = Model.flower_model
        model.load_state_dict(torch.load(flower_weight_path))

        # 数据库创建表语句
        sql_create_table = """CREATE TABLE FLOWER
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        image BLOB,
        feature_code BLOB,
        category STRING)"""
    else:
        model = Model.car_model
        model.load_state_dict(torch.load(car_weight_path))

        sql_create_table = """CREATE TABLE CAR
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB,
                feature_code BLOB,
                category STRING)"""

    model.fc = nn.Identity()
    model.eval()

    # 创造数据库表并插入
    coon = sqlite3.connect(db)
    cursor = coon.cursor()
    # cursor.execute(sql_create_table)

    image_list = os.listdir(dir_path)
    for i in image_list:
        # 获取图像类别
        if tb == "flower":
            category = i.split('_')[-1].split('.')[0]
        else:
            category, ext = os.path.splitext(i.split('_', 1)[1])

        # 获取图像二进制编码
        image_path = os.path.join(dir_path, i)
        image = Image.open(image_path).convert("RGB")
        binary_data = image.tobytes()
        # 获取图像二级制特征编码
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature_code = model(image_tensor)
        tensor_bytes = io.BytesIO()
        torch.save(feature_code, tensor_bytes)
        tensor_bytes = tensor_bytes.getvalue()

        # 数据库插入语句
        cursor.execute("""INSERT INTO FLOWER (image, feature_code, category) VALUES (?, ?, ?)""",
                       (binary_data, tensor_bytes, category)
                       )
    coon.commit()
    coon.close()


if __name__ == '__main__':
    # sql_create_table = """CREATE TABLE FLOWER
    #         (id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         image BLOB,
    #         feature_code BLOB,
    #         category STRING)"""
    # coon = sqlite3.connect(db_path)
    # cursor = coon.cursor()
    #
    # cursor.execute(sql_create_table)
    # cursor.execute("""INSERT INTO FLOWER (image, feature_code, category) VALUES (?, ?, ?)""",
    #                (None, None, None)
    #                )
    create_table(db_path, "car", car_database_path)
