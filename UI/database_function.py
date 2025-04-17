import sqlite3
import io
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import Retrival.model as Model
import torch.nn as nn
import torch.nn.functional as F


# 你需要将一下路径更改为你自己的路径
db_path = "你自己的数据库路径"
flower_weight_path = "你自己的花模型权重路径"
# Image.oepn打开的图片在数据格式上和cv2打开的有所不同
test_image = Image.open("你自己的测试图片路径").convert("RGB")


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_tensor = transform(test_image).unsqueeze(0)
model = Model.flower_model
model.load_state_dict(torch.load(flower_weight_path))
feature_model = model
feature_model.fc = nn.Identity()
feature_model.eval()
with torch.no_grad():
    base_tensor = feature_model(image_tensor)


# 查找数据库中的feature_code并返回
def get_feature_code(db, tb):
    coon = sqlite3.connect(db)
    cursor = coon.cursor()

    # 数据库存储图像分为车和花存储在不同的表中， 手动传入表明进行存储
    if tb == "FLOWER":
        cursor.execute("""SELECT id, feature_code FROM FLOWER""")
        result = cursor.fetchall()
        similarity_list = []
        for row in result:
            id, feature_bytes = row
            
            feature_code_tensor = torch.load(io.BytesIO(feature_bytes))
            similarity = F.cosine_similarity(base_tensor, feature_code_tensor)
            similarity_list.append([id, similarity.item()])

        sorter_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
        for i in sorter_list:
            id, similar = i
            print(id, similar)


def Insert_image():
    None


if __name__ == '__main__':
    get_feature_code(db_path, "FLOWER")
