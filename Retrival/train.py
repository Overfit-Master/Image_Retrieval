import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from  torch.optim.lr_scheduler import StepLR

import data_process
import model

# 参数配置
config = {
    "batch_size": 32,
    "lr_rate": 0.0001,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # "device": torch.device("cpu"),
    "flower": model.flower_model,
    "car": model.car_model,
    "epochs": 100,
    "weight_dir_path": r"C:\Users\31825\Desktop\diploma_project\Retrival\weight",
    "log_dir_path": r"C:\Users\31825\Desktop\diploma_project\Retrival\log"
}


def train(task: str, epochs):
    net = config[task]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=config['lr_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    train_data = DataLoader(data_process.My_Dataset(task, "all"), batch_size=config["batch_size"], shuffle=True,
                            num_workers=2)
    test_data = DataLoader(data_process.My_Dataset(task, "retrival"), batch_size=config["batch_size"], shuffle=False,
                           num_workers=2)

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M')

    best_pt_file = task + '-' + formatted_time + '_best.pt'
    best_pt_path = os.path.join(config["weight_dir_path"], best_pt_file)
    last_pt_file = task + '-' + formatted_time + '_last.pt'
    last_pt_path = os.path.join(config["weight_dir_path"], last_pt_file)
    log_file = task + '_' + formatted_time + '.txt'
    log_path = os.path.join(config["log_dir_path"], log_file)
    test_file = task + '_' + formatted_time + '_test.txt'
    test_path = os.path.join(config["log_dir_path"], test_file)

    print("Model training start at {}".format(current_time))
    print("Train on {}".format(config["device"]))

    best_acc = 0
    for epoch in range(epochs):
        current_time = time.strftime('%H:%M::%S', time.localtime(time.time()))
        print("%d--[%s] is training..." % (epoch + 1, current_time))

        # 模型训练
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>TRAINING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        net.train()
        net.to(config["device"])

        total_train_loss = 0
        total_train_correct = 0
        train_num = 0

        for img, label, _ in tqdm(train_data):
            img = img.to(config["device"])
            label = label.to(config["device"])

            predict = net(img)
            loss = loss_fn(predict, label)
            total_train_loss += loss.item()
            correct_num = (predict.argmax(dim=1) == label).sum().item()
            total_train_correct += correct_num
            train_num += len(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_acc = total_train_correct / train_num
        with open(log_path, "a") as f:
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
                    img = img.to(config['device'])
                    label = label.to(config['device'])

                    predict = net(img)
                    loss = loss_fn(predict, label)

                    total_test_loss += loss.item()
                    correct_num = (predict.argmax(dim=1) == label).sum().item()
                    total_test_correct += correct_num
                    test_num += len(img)

                accuracy = total_test_correct / test_num
                with open(test_path, "a") as f:
                    f.write("epoch {}--test:{}, loss:{}\n".format(epoch + 1, accuracy, total_test_loss))
                    f.close()

                print("Test loss is: {}".format(total_test_loss))
                print("Test acc is: {}".format(accuracy))

                if accuracy > best_acc:
                    torch.save(net.state_dict(), best_pt_path)

                if epoch + 1 == config["epochs"]:
                    torch.save(net.state_dict(), last_pt_path)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    train("car", config["epochs"])
