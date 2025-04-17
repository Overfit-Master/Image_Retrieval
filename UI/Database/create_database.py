import os
import sqlite3
import torch.nn as nn
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import io
from tqdm import tqdm

import Retrival.model as Model

database_path = './retrival_database.db'
flower_root_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Oxford_flowers-17\Divide_data\database"
car_root_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Stanford_Cars\Divide_data\database"
flower_weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\flower-2024-05-16-19-31_best.pt"
car_weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\car-2024-05-17-13-03_best.pt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 检查表明是否在数据库中
def check(db, tb_name):
    cursor_temp = db.cursor()  # 类似光标？
    sql = """SELECT tbl_name FROM sqlite_master WHERE type = 'table'"""  # sql查询语句
    cursor_temp.execute(sql)
    values = cursor_temp.fetchall()
    tables = []
    for v in values:
        tables.append(v[0])
    if tb_name not in tables:
        return True
    else:
        print("False")
        return False


def insert(db_cur, tb, root_path):
    file_name = os.listdir(root_path)

    # 载入网络模型
    if tb == "FLOWER":
        model = Model.flower_model
        model.load_state_dict(torch.load(flower_weight_path))
    else:
        model = Model.car_model
        model.load_state_dict(torch.load(car_weight_path))
    # 此处不需要计算类别，去除全连接层
    feature_model = model
    feature_model.fc = nn.Identity()
    feature_model.eval()

    for i in tqdm(file_name):
        path = os.path.join(root_path, i)
        if tb == "FLOWER":
            category = i.split('_')[-1].split('.')[0]
        else:
            category, ext = os.path.splitext(i.split('_', 1)[1])

        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # 模型的原始输入含有batch_size，为四维数据，此处进行升维操作

        with torch.no_grad():
            feature_code = feature_model(image_tensor)
        # 将torch.tensor格式的数据直接转为二级制比特流存储
        tensor_bytes = io.BytesIO()
        torch.save(feature_code, tensor_bytes)
        tensor_bytes = tensor_bytes.getvalue()

        # todo 将path,category,feature_code插入数据库
        if tb == "FLOWER":
            db_cur.execute("""INSERT INTO FLOWER (image_path, feature_code, category) VALUES (?, ?, ?)""",
                           (path, tensor_bytes, category)
                           )
        else:
            db_cur.execute("""INSERT INTO CAR (image_path, feature_code, category) VALUES (?, ?, ?)""",
                           (path, tensor_bytes, category)
                           )
    coon.commit()
    coon.close()


if __name__ == '__main__':
    coon = sqlite3.connect(database_path)
    cursor = coon.cursor()
    # check(coon, "FLOWER")
    # if not check(coon, "FLOWER"):
    #     sql_create_table = """CREATE TABLE FLOWER
    #     (id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     image_path STRING,
    #     feature_code BLOB,
    #     category STRING);
    #     """
    #     cursor.execute(sql_create_table)

    file_name = os.listdir(flower_root_path)
    model = Model.flower_model
    model.load_state_dict(torch.load(flower_weight_path))

    # 此处不需要计算类别，去除全连接层
    feature_model = model
    feature_model.fc = nn.Identity()
    feature_model.eval()

    sql_create_table = """CREATE TABLE FLOWER
                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   image_path STRING,
                   feature_code BLOB,
                   category STRING);
                   """
    cursor.execute(sql_create_table)


    for i in tqdm(file_name):
        path = os.path.join(flower_root_path, i)
        category = i.split('_')[-1].split('.')[0]
        # category, ext = os.path.splitext(i.split('_', 1)[1])      # 车的类别分割

        image = Image.open(path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # 模型的原始输入含有batch_size，为四维数据，此处进行升维操作

        with torch.no_grad():
            feature_code = feature_model(image_tensor)
        # 将torch.tensor格式的数据直接转为二级制比特流存储
        tensor_bytes = io.BytesIO()
        torch.save(feature_code, tensor_bytes)
        tensor_bytes = tensor_bytes.getvalue()

        # todo 将path,category,feature_code插入数据库
        cursor.execute("""INSERT INTO FLOWER (image_path, feature_code, category) VALUES (?, ?, ?)""",
                       (path, tensor_bytes, category)
                       )
        #   cursor.execute("""INSERT INTO CAR (image_path, feature_code, category) VALUES (?, ?, ?)""",
        #                    (path, tensor_bytes, category)
        #                    )
    coon.commit()
    coon.close()
