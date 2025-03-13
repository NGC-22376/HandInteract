
import torch
import torch.nn as nn
import torch.optim as optim
from utils.window import *
from torch.utils.data import DataLoader, Dataset


# 转换为TensorDataset

class cnn(nn.Module):
    def __init__(self, num_classes):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)  # (64,8,128,4)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)  # (64,8,64,2)4+5+2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)  # (64,16,64,2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 1))  # (64,16,32,2)5+3+1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # (64,32,32,2)
        # self.pool3 = nn.MaxPool2d((2,1))#(64,64,8,2)
        self.bn3 = nn.BatchNorm2d(32)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (64,32,1,1)
        self.dropout = nn.Dropout(0.5)
        '''
        self.fc1 = nn.Linear(2560, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        '''
        self.fc1 = nn.Linear(32, num_classes)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        # print("1.1",x.shape)
        x = self.pool1(x)
        # print("1.2",x.shape)
        x = torch.relu(self.bn2(self.conv2(x)))
        # print("2.1",x.shape)
        x = self.pool2(x)
        # print("2.2",x.shape)
        x = torch.relu(self.bn3(self.conv3(x)))
        # print("3.1",x.shape)
        # x = self.pool3(x)
        # print("3.2",x.shape)
        x = self.avg_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)  # (64, 512)
        x = self.fc1(x)
        # x = self.flat(x)
        # print("x.shape",x.shape)
        # x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = self.fc4(x)
        return x


class WindowDataset(Dataset):
    def __init__(self, windows, labels, augment=True):
        self.windows = windows
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 8
    subject_num = 1
    x_all, y_all = [[] for _ in range(subject_num)], [[] for _ in range(subject_num)]
    data_split_point_all, datanum_all = [[] for _ in range(subject_num)], [[] for _ in range(subject_num)]
    for s in range(subject_num):
        print("Loading...S" + str(s + 1))
        x_all[s], y_all[s], data_split_point_all[s], datanum_all[s] = get_feature_window(s)  # x_all(用户数，分类数8，窗口数，1，窗口长128，特征数4)y_all(用户数，分类数8，窗口数4)
        # print("d",y_all[s])

    x1_a, y1_a, x1_b, y1_b = result_intra(x_all, y_all, datanum_all)
    batch_size = 64
    total_testaccuracy = 0
    mean_testaccuracy = 0

    for subject in range(subject_num):

        train_dataset = WindowDataset(x1_a[subject], y1_a[subject])
        valid_dataset = WindowDataset(x1_b[subject], y1_b[subject])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        # 模型初始化
        model = cnn(num_classes=8)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        model.load_state_dict(torch.load(f"semg4(2)newbest_model_S0.pth"))#加载4通道semg的self.pool1 = nn.MaxPool2d(2)的模型参数
        #    print("已加载模型参数")

        epoch = 40
        best_val_acc = 0
        patience = 5
        counter = 0

        for i in range(epoch):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            # print("第{}轮训练".format(i+1))

            # 模型训练
            for x1a, y1a in train_loader:  # 加载数据x1a(64,1,128,4)
                x1a,y1a = x1a.to(device),y1a.to(device)
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(x1a)
                loss = criterion(output, y1a)

                loss.backward()  # 打开梯度
                optimizer.step()  # 更新模型参数

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y1a.size(0)
                correct += predicted.eq(y1a).sum().item()
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"{subject}Epoch [{i + 1}/{epoch}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # 模型测试
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():  # 保证跑测试的时候不影响梯度
                for x1b, y1b in valid_loader:
                    x1b,y1b = x1b.to(device),y1b.to(device)
                    output = model(x1b)
                    _, predicted = torch.max(output, 1)
                    total += y1b.size(0)
                    correct += (predicted == y1b).sum().item()

            test_accuracy = 100 * correct / total
            print(f"{subject}Test Accuracy: {test_accuracy:.2f}%")
            if test_accuracy > best_val_acc:
                best_val_acc = test_accuracy
                counter = 0
                torch.save(model.state_dict(), 'best_model_S' + str(subject) + '.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"{subject}Best Accuracy: {best_val_acc:.2f}%")
                    total_testaccuracy += best_val_acc
                    mean_testaccuracy = total_testaccuracy / (subject + 1)
                    print(f"Early stopping at epoch {i + 1}")
                    break
            scheduler.step(test_accuracy)
        print(f"#########mean_testaccuracy:{mean_testaccuracy:.2f}######")
