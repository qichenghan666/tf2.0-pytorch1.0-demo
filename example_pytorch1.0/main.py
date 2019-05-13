import torch
print(torch.__version__)
print(torch.cuda.is_available())
import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import functional as F
from torch import optim
from visdom import Visdom

# 自定义数据集 ：二分类
class myDataset(Dataset):
    def __init__(self, class1_dir, class2_dir):
        self.x, self.y = self.preprocess(class1_dir, class2_dir)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

    def _read_data(self, dataset_dir):
        data_dir = os.path.join('./dataset/', dataset_dir)
        data_dir = data_dir + '/*.jpg'
        eye_data = glob.glob(data_dir)
        x = []
        for img in eye_data:
            img = Image.open(img)
            img = np.asarray(img)
            x.append(img)
        x = np.array(x)
        x = torch.from_numpy(x).float() / 255.
        x = torch.unsqueeze(x, dim=3)
        return x

    # 预处理
    def preprocess(self, class1_dir, class2_dir):
        x1 = self._read_data(class1_dir)
        y1 = torch.ones(x1.shape[0])

        x2 = self._read_data(class2_dir)
        y2 = torch.zeros(x2.shape[0])

        x = torch.cat([x1, x2], dim=0)
        y = torch.cat([y1, y2], dim=0)
        return x, y


train_dataset = myDataset('closedEyes', 'openEyes')
test_dataset = myDataset('close_test', 'open_test')
# print(train_x.type(), train_y.type(), test_x.type(), test_y.type())

train_db = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_db = DataLoader(test_dataset, batch_size=64, shuffle=True)

# x, y = iter(train_db).next()
# print(x.shape, y.shape)

class Basenet(nn.Module):
    def __init__(self, input, output):
        super(Basenet, self).__init__()

        self.conv1 = nn.Conv2d(input, 64, kernel_size=[3, 3], padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=[3, 3], padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3 = nn.Conv2d(64, 128, kernel_size=[3, 3], padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=[3, 3], padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=[3, 3], padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=[3, 3], padding=1)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=[3, 3], padding=1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512, 512, kernel_size=[3, 3], padding=1)
        self.relu8 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.dense1 = nn.Linear(512, 256)
        self.relu9 = nn.ReLU()
        self.dense2 = nn.Linear(256, 128)
        self.relu10 = nn.ReLU()
        self.dense3 = nn.Linear(128, output)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)

        x = self.relu4(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.maxpool4(x)

        x = torch.reshape(x, [-1, 512])
        x = self.dense1(x)
        x = self.relu9(x)
        x = self.dense2(x)
        x = self.relu10(x)
        x = self.dense3(x)


        return x


device = torch.device('cuda')
basenet = Basenet(24, 2).to(device)
# x, _ = iter(train_db).next()
# test = basenet(x.to(device))
# print(basenet)

criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(basenet.parameters(), lr=1e-4)

viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))

globals_step = 0
for epoch in range(10):
    # train mode
    basenet.train()
    for step, (x, y) in enumerate(train_db):
        x, y = x.to(device), y.to(device)
        # forward
        logit = basenet(x)
        # loss
        # print(logit.type(), y.type())
        # logit = logit.long()
        loss = criteon(logit, y.long())
        # grads
        optimizer.zero_grad()
        loss.backward()
        # update
        optimizer.step()
        if step % 10 == 0:
            print('epoch:', epoch, 'loss:', loss.item())

        globals_step += 1
        viz.line([loss.item()], [globals_step], win='train_loss', update='append')

    # turn to eval mode
    basenet.eval()
    with torch.no_grad():
        total_num = 0
        total_correct = 0
        for x, y in test_db:
            x, y = x.to(device), y.to(device)
            logit = basenet(x)
            prob = F.softmax(logit, dim=1)
            pred = torch.argmax(prob, dim=1)
            correct = torch.eq(pred, y.long()).sum().item()

            total_num += x.shape[0]
            total_correct +=correct
        acc = total_correct / total_num
        print('epoch:', epoch, 'acc:', acc)

torch.save(basenet.state_dict(), 'eyes.pkl')

del basenet

basenet = Basenet(24, 2).to(device)
basenet.load_state_dict(torch.load('eyes.pkl'))

basenet.eval()
with torch.no_grad():
    total_num = 0
    total_correct = 0
    for x, y in test_db:
        x, y = x.to(device), y.to(device)
        logit = basenet(x)
        prob = F.softmax(logit, dim=1)
        pred = torch.argmax(prob, dim=1)
        correct = torch.eq(pred, y.long()).sum().item()

        total_num += x.shape[0]
        total_correct += correct
    acc = total_correct / total_num
    print('epoch:', epoch, 'acc:', acc)

params = basenet.state_dict()
for k, v in params.items():
    print(k)  # 打印网络中的变量名
print(params['conv1.weight'])  # 打印conv1的weight
print(params['conv1.bias'])  # 打印conv1的bias
