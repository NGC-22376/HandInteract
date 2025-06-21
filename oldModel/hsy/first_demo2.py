# 1通道复制为4通道
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from utils.window import *
from torch.utils.data import DataLoader, Dataset


class cnn(nn.Module):
    def __init__(self, num_classes):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1, padding=1)  # (64,8,128,10)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)  # (64,8,64,5)4+5+2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=1)  # (64,16,64,5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 1))  # (64,16,32,5)5+3+1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1)  # (64,32,32,5)
        self.bn3 = nn.BatchNorm2d(32)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (64,64,1,1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32, num_classes)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        feature = x
        x = self.fc1(x)
        return x,feature


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


def Mean_Template_Matching(subject, model, x_train, y_train, tx_test, ty_test):  # train模板提取，test最终评估,split(72=8*9)
    detection_win_size = 35  # 投票长度
    model.eval()

    template_label = [[] for _ in range(classes)]
    right_out = [[] for _ in range(classes)]
    for i in range(len(y_train)):
        template_label[y_train[i]].append(i)

    update = 0
    tx_test = torch.from_numpy(tx_test).to(device, dtype=torch.float)

    model.eval()

    # 计算模板均值
    template_output_mean = []

    for n in range(classes):
        use_data = torch.from_numpy(np.array(x_train[template_label[n]])).to(device, dtype=torch.float)
        _, template_output = model(use_data)
        template_output = template_output.cpu().detach().numpy()  # (23,128)
        template_output_mean.append(template_output.mean(axis=0))  # 训练出的均值模板(8,128)#加lda后（8,7）
    template_output_mean_old = template_output_mean.copy()


    # 初始化队列和窗口计数器
    window_queue_new = deque()
    fixed_win_new = [0] * classes
    window_queue_old = deque()
    fixed_win_old = [0] * classes
    right_new, right_old, vote_right_new, vote_right_old = 0, 0, 0, 0
    x_right, y_right, index = [], [], []
    yu_out=[[] for _ in range(classes)]
    for m in range(
            len(tx_test)):  # tx_test=(1785,1,32,10),tx_test_split=(178,1,32,10),test_output_split=(178,128),test_out=(1785,128)
        sample = tx_test[m]
        # sample = torch.from_numpy(np.array(sample)).cpu()
        sample = torch.unsqueeze(sample, 0).to(device)
        _, sample_feature = model(sample)
        sample_feature = sample_feature.cpu().detach().numpy()
        sample_feature = sample_feature.tolist()  # (89,128)

        # 计算输出类别、置信度

        yu = 0.7

        sample_feature_normalized = sample_feature / np.linalg.norm(sample_feature, axis=1, keepdims=True)  # (89,128)
        template_output_mean_normalized = template_output_mean / np.linalg.norm(template_output_mean, axis=1,
                                                                                keepdims=True)  # (8,128)
        template_output_mean_old_normalized = template_output_mean_old / np.linalg.norm(template_output_mean_old,
                                                                                        axis=1,
                                                                                        keepdims=True)  # (8,128)
        similarities_new, similarities_old = [], []
        for c in range(classes):
            L1_new = np.sum(np.abs(sample_feature_normalized - template_output_mean_normalized[c]))
            L1_old = np.sum(np.abs(sample_feature_normalized - template_output_mean_old_normalized[c]))
            similarities_new.append(L1_new)
            similarities_old.append(L1_old)

        sorted_indices = np.argsort(np.array(similarities_new).flatten())  # 降序排列索引
        test_label1st = similarities_new[sorted_indices[0]]
        test_label2nd = similarities_new[sorted_indices[1]]
        outlabel_new = np.argmin(similarities_new)
        outlabel_new_sus = sorted_indices[1]
        outlabel_old = np.argmin(similarities_old)

        if ty_test[m]==outlabel_new:
            yu_out[ty_test[m]].append(test_label1st/test_label2nd)
            right_out[ty_test[m]].append(1)
        else:
            yu_out[ty_test[m]].append(test_label1st/similarities_new[ty_test[m]])
            right_out[ty_test[m]].append(0)

        if test_label1st / test_label2nd < yu:
            update = 1
            cpu_data = sample.cpu().numpy()
            x_right.append(cpu_data)
            y_right.append(outlabel_new)

        # 更新模板均值
        if update == 1:


            template_output_mean = []
            for n in range(classes):
                use_data = torch.from_numpy(np.array(x_train[template_label[n]])).to(device, dtype=torch.float)
                for l in range(len(x_right)):
                    if y_right[l] == n:
                        x_right_tensor = torch.from_numpy(np.array(x_right[l])).to(device)
                        use_data = torch.cat((use_data, x_right_tensor), dim=0).to(device)
                _, template_output = model(use_data)
                template_output = template_output.cpu().detach().numpy()  # (23,128)
                template_output_mean.append(template_output.mean(axis=0))  # 训练出的均值模板(8,128)#加lda后（8,7）
                # print(template_output_mean[n][0],template_output_mean_old[n][0])

        # 投票
        if m > 0:
            if ty_test[m] != ty_test[m - 1]:
                window_queue_new.clear()
                fixed_win_new = [0] * classes

        if outlabel_new == ty_test[m]:
            right_new += 1

        if update == 1 and outlabel_new != ty_test[m]:
            print(m, outlabel_new, outlabel_new_sus, test_label1st, test_label2nd, ty_test[m])

        # 更新窗口
        current_label = outlabel_new
        window_queue_new.append(current_label)
        fixed_win_new[current_label] += 1

        # 维护窗口长度不超过 detection_win_size
        while len(window_queue_new) > detection_win_size:
            oldest_label = window_queue_new.popleft()
            fixed_win_new[oldest_label] -= 1

            # 投票决策
        if fixed_win_new.index(max(fixed_win_new)) == ty_test[m]:
            vote_right_new += 1


        test_accuracy_new = right_new / (m + 1)
        vote_test_accuracy_new = vote_right_new / (m + 1)

        update = 0

        # 旧模板投票
        if m > 0:
            if ty_test[m] != ty_test[m - 1]:
                window_queue_old.clear()
                fixed_win_old = [0] * classes

        if outlabel_old == ty_test[m]:
            right_old += 1

        # 更新窗口
        current_label = outlabel_old
        window_queue_old.append(current_label)
        fixed_win_old[current_label] += 1

        # 维护窗口长度不超过 detection_win_size
        while len(window_queue_old) > detection_win_size:
            oldest_label = window_queue_old.popleft()
            fixed_win_old[oldest_label] -= 1

            # 投票决策
        if fixed_win_old.index(max(fixed_win_old)) == ty_test[m]:
            vote_right_old += 1

        test_accuracy_old = right_old / (m + 1)
        vote_test_accuracy_old = vote_right_old / (m + 1)


    if test_accuracy_new > test_accuracy_old or vote_test_accuracy_new > vote_test_accuracy_old:
        print(f"conguradulation!!")
    print(f"test_accuracy_new:{test_accuracy_new},test_accuracy_old:{test_accuracy_old}")
    print(f"vote_test_accuracy_new:{vote_test_accuracy_new},vote_test_accuracy_old:{vote_test_accuracy_old}")
    print(len(y_right))
    return test_accuracy_new, vote_test_accuracy_new, test_accuracy_old, vote_test_accuracy_old,yu_out,right_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 8
    subject_num = 1
    x_all, y_all = [[] for _ in range(subject_num)], [[] for _ in range(subject_num)]
    data_split_point_all, datanum_all = [[] for _ in range(subject_num)], [[] for _ in range(subject_num)]
    for s in range(subject_num):
        print("Loading...S" + str(s + 1))
        x_all[s], y_all[s], data_split_point_all[s], datanum_all[s] = get_feature_window(
            s)  # x_all(用户数27，分类数16，窗口数，1，窗口长32，特征数10)y_all(用户数27，分类数16，窗口数)data_split_point_all暂时用处不大，每个手势的文件数不同
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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        model.load_state_dict(torch.load(r"D:\WeChatfiles\WeChat Files\wxid_u266ve7902wp22\FileStorage\File\2025-02\HandIteract\src\newbest_model_S0.pth"))

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
            for x1a, y1a in train_loader:  # 加载数据x1a(64,1,64,1)
                optimizer.zero_grad()  # 优化器梯度清零
                output,_ = model(x1a)
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
                    output,_ = model(x1b)
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

        sorted_yu = [[] for _ in range(classes)]
        #model.load_state_dict(torch.load('best_model_S0.pth'))
        test_accuracy_new, vote_test_accuracy_new, test_accuracy_old, vote_test_accuracy_old,yu_out,right_out = Mean_Template_Matching(subject,model, x1_a[subject], y1_a[subject], x1_b[subject],y1_b[subject])
        
        print("------------------------------------------------------")
        print(f"accuracy:{test_accuracy_new * 100:.2f}%,vote_accuracy:{vote_test_accuracy_new * 100:.2f}%")
        print("------------------------------------------------------")
     
