import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import time

train_txt_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Classify\train.txt"
test_txt_path = r"C:\Users\31825\Desktop\diploma_project\Retrival\Datasets\Classify\test.txt"

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)


def get_path_list(txt_path):
    path_list = []
    label_list = []
    with open(txt_path, "r") as f:
        for msg in f.readlines():
            path_list.append(msg.split(',')[0])
            label_list.append(msg.split(',')[1][:-1])
    return path_list, label_list


class MyDataset(Dataset):
    def __init__(self, trans, txt_path):
        self.transform = trans
        path_list, label_list = get_path_list(txt_path)
        self.path_list = path_list
        self.label_list = label_list

    def __getitem__(self, index):
        img = self.path_list[index]
        label = int(self.label_list[index])

        img = Image.open(img).convert('RGB')
        img = self.transform(img)

        label = torch.tensor(label)  # 图片正向传播计算出的预测值为tensor，与图片的标签作对比后进行反向传播，所以标签也要为tensor

        return img, label, index

    def __len__(self):
        return len(self.path_list)


train = MyDataset(transform, train_txt_path)
test = MyDataset(transform, test_txt_path)
train_data = DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
test_data = DataLoader(test, batch_size=32, shuffle=False, num_workers=2)
net = model.classify_model
epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.0001)

if __name__ == '__main__':
    best_acc = 0
    for epoch in range(epochs):
        current_time = time.strftime('%H:%M::%S', time.localtime(time.time()))
        print("%d--[%s] is training..." % (epoch + 1, current_time))

        # 模型训练
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        net.train()
        net.to("cuda:0")

        total_train_loss = 0
        total_train_correct = 0
        train_num = 0

        for img, label, _ in tqdm(train_data):
            img = img.to("cuda:0")
            label = label.to("cuda:0")

            predict = net(img)
            loss = loss_fn(predict, label)
            total_train_loss += loss.item()
            correct_num = (predict.argmax(dim=1) == label).sum().item()
            total_train_correct += correct_num
            train_num += len(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = total_train_correct / train_num
        with open("./log/classify_train.txt", "a") as f:
            f.write("epoch {}--train:{}, loss:{}\n".format(epoch + 1, train_acc, total_train_loss))
            f.close()

        print("Train loss is: {}".format(total_train_loss))
        print("Train acc is: {}".format(train_acc))

        # 模型测试
        if (epoch + 1) % 5 == 0:
            net.eval()
            total_test_loss = 0
            total_test_correct = 0
            test_num = 0

            with torch.no_grad():
                for img, label, _ in tqdm(test_data):
                    img = img.to("cuda:0")
                    label = label.to("cuda:0")

                    predict = net(img)
                    loss = loss_fn(predict, label)

                    total_test_loss += loss.item()
                    correct_num = (predict.argmax(dim=1) == label).sum().item()
                    total_test_correct += correct_num
                    test_num += len(img)

                accuracy = total_test_correct / test_num
                with open("./log/classify_test.txt", "a") as f:
                    f.write("epoch {}--test:{}, loss:{}\n".format(epoch + 1, accuracy, total_test_loss))
                    f.close()

                print("Test loss is: {}".format(total_test_loss))
                print("Test ass is: {}".format(accuracy))

                if accuracy > best_acc:
                    torch.save(net.state_dict(), "./weight/best_classify.pt")

                if epoch + 1 == epochs:
                    torch.save(net.state_dict(), "./weight/last_classify.pt")
