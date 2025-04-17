import sqlite3
import io
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import Retrival.model as Model
import torch.nn as nn
import torch.nn.functional as F

db_path = r"./Database/retrival_database.db"
flower_weight_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\weight\flower-2024-03-31-16-55_best.pt"
test_image = Image.open(r"C:\Users\31825\Desktop\diploma_project\UI\test_image\image_0241.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)
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
