import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.keep_prob = 0.7

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = torch.nn.Linear(31 * 31 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        
        self.fc2 = torch.nn.Linear(625, 4, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        
        return out


def get_train():
    train_path = './data/train'

    transform = transforms.Compose(
        [
            transforms.RandomRotation(20),  # 이미지를 최대 20도 회전
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 색상에 변화를 줌
            transforms.RandomHorizontalFlip(),  # 50% 확률로 수평 뒤집기
            transforms.RandomResizedCrop(size=(244, 244), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 임의로 크롭 후 크기 조절
            transforms.ToTensor(),  # PIL 이미지 또는 NumPy ndarray를 PyTorch 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        train_path,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    return train_loader


def get_test():
    test_path = './data/val'

    transform = transforms.Compose(
        [
            transforms.Resize((244, 244)),  # 테스트 이미지를 244x244 크기로 리사이징
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        test_path,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True
    )

    return test_loader


def train(training_epochs, train_loader, device, optimizer, model, criterion, total_batch):
    train_acc_history  = []
    train_loss_history = []

    for epoch in range(training_epochs):
        avg_cost = 0
        correct_predictions = 0
        total_predictions = 0

        for X, Y in train_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

            # 정확도 계산
            _, predicted = torch.max(hypothesis, 1)  # 가장 높은 확률을 가진 클래스를 예측값으로 선택
            correct_predictions += (predicted == Y).sum().item()  # 예측이 맞은 개수를 누적
            total_predictions += Y.size(0)  # 전체 예측 개수를 누적

        avg_cost = avg_cost / total_batch
        accuracy = correct_predictions / total_predictions * 100  # 정확도 계산

        train_acc_history.append(accuracy)
        train_loss_history.append(avg_cost)

        print('[Epoch: {:>4}] cost = {:>.9}, Accuracy = {:>.2f}%'.format(epoch + 1, avg_cost, accuracy))

    return train_acc_history


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate   = 0.001
    training_epochs = 50

    train_loader = get_train()
    test_loader  = get_test()

    total_samples = len(train_loader.dataset)  # 데이터셋의 총 샘플 수
    batch_size    = train_loader.batch_size  # 배치 사이즈
    total_batch   = math.ceil(total_samples / batch_size)  # 총 배치 수, math.ceil로 올림 처리


    model     = CNN().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_acc_history = train(training_epochs, train_loader, device, \
                              optimizer, model, criterion, total_batch)
    
    test(model, test_loader, device)

    plt.plot(train_acc_history)
    plt.savefig('./result.png')


if __name__ == '__main__':
    main()