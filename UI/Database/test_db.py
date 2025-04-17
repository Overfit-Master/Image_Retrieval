import sqlite3
import torch.nn as nn
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import io

import Retrival.model as Model

model = Model.flower_model
weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\flower-2024-03-31-16-55_best.pt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]
                         )
]
)

model.load_state_dict(torch.load(weight_path))
feature_model = model
feature_model.fc = nn.Identity()
feature_model.eval()

path = "../test_image/image_0241.jpg"
input_image = Image.open('../test_image/image_0241.jpg').convert("RGB")
image_tensor = transform(input_image).unsqueeze(0)

with torch.no_grad():
    feature_code = feature_model(image_tensor)

print(feature_code)

tensor_bytes = io.BytesIO()
torch.save(feature_code, tensor_bytes)
tensor_bytes = tensor_bytes.getvalue()

coon = sqlite3.connect('retrival_database.db')
cursor = coon.cursor()

sql_create_table = """CREATE TABLE FLOWER
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        feature_code BLOB,
        category TEXT);
        """
cursor.execute(sql_create_table)

cursor.execute("""INSERT INTO FLOWER (image_path, feature_code, category) VALUES (?, ?, ?)""", (path, tensor_bytes, "flower"))
coon.commit()   # 提交执行~保存

cursor.execute("""SELECT feature_code FROM FLOWER WHERE image_path = ?""", (path,))
# print(cursor.fetchone())
feature_code_bytes = cursor.fetchone()[0]       # fetchone获取image_path匹配的这一行数据，除去image_path这一列，feature_code为第一个，所以通过索引[0]访问

feature_code_tensor = torch.load(io.BytesIO(feature_code_bytes))
print(feature_code_tensor)
