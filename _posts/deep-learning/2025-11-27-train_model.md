---
layout: post
title: "train model"
date:   2025-11-27
tags: [deep learning]
comments: true
author: zhw
mathjax: true
---


# 训练模型

训练模型我们一般分为:  加载数据, 实现网络， 寻找损失函数，优化器，计算精确值，训练，在测试集上测试。

## 加载数据

我们使用datasets 来下载训练和测试数据集，并使用transforms来转换成tensor的结构，最后返回DataLoader的批量数据。

```python
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

def load_data(batch_size , resize):
    trans = transforms.Compose([
    transforms.ToTensor(),
    ])
    if resize:
        trans.transforms.insert(0, transforms.Resize((resize, resize)))

    mnist_train = datasets.FashionMNIST("./dataset", train=True, download=True, transform=trans)
    mnist_test = datasets.FashionMNIST("./dataset", train=False, download=True, transform=trans)

    return (DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

```



如果我们的数据集是图像，我们可以使用matplot 来打印出图像的内容:

```python
def print_image():
    train_loader, test_loader = load_data(64, 64)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 8))
    for i in range(12):
        plt.subplot(2, 6, i+1)
        plt.imshow(images[i].numpy().reshape(28, 28))
        plt.title(f"{classes[labels[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class Animator:
    def __init__(self, xlabel = None, ylabel = None, legend = None,
                 xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
                 fmts = ('-', 'm--', 'g-.', 'r:'), nrows = 1, ncols = 1,
                 figsize = (3.5, 2.5)):
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        if legend:
            axes.legend(legend)
        axes.grid()
    
    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y , fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

## 实现网络

这里我们以LeNet 为例: 使用nn.Sequential 来把所有的网络都放在一起，包括卷积层，pooling层，和全连接层。

```python
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(256*5*5, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(4096, 10)
)

X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
     print(layer.__class__.__name__, 'output shape:\t', X.shape)
```



## 训练

```python
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def accuracy(y_hat, y):
    """返回预测正确的样本个数（float 类型）"""
    # 如果 y_hat 是 logits 或概率（如 [N, C]），取预测类别
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, dim=1) 
    
    # 比较预测 vs 真实标签，并确保类型一致（避免 int64 vs int32 问题）
    cmp = (y_hat.to(y.dtype) == y)
    
    # 求 cmp 中 True 的个数（True=1, False=0）
    return float(cmp.sum())  # cmp.sum() 等价于 torch.sum(cmp)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]



# train_loader, test_loader = load_data(128, 224)
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)

def train_and_eval(net, batch_size = 128, resize = 224, lr=0.05, num_epochs=10):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    train_loader, test_loader = load_data(batch_size, resize)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    total_start = time.time()
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()

        epoch_start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss_value * images.shape[0], accuracy(outputs, labels), images.shape[0])

            epoch_time = time.time() - epoch_start
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if ((i+1) % (num_batches // 5)) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_loader, device)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch {epoch + 1}, '
            f'loss {train_l:.3f}, '
            f'train acc {train_acc:.3f}, '
            f'test acc {test_acc:.3f}, '
            f'time {epoch_time:.2f}s'
            f'on {str(device)}')
    total_time = time.time() - total_start
    total_samples = metric[2] * num_epochs

    print(f'Total training time: {total_time:.2f}s')
    print(f'{total_samples / total_time:.1f} examples/sec on {device}')
```



