# **第五章PyTorch模型定义**

参考：

[pytorch模型定义]: https://github.com/datawhalechina/thorough-pytorch/tree/main/%E7%AC%AC%E4%BA%94%E7%AB%A0%20PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89

## 一：pytorch 模型定义的方式

### 学习目标：

熟悉pytorch中模型定义的三种方式

读懂github上千奇百怪的写法

自己根据需要灵活选取模型定义的方式

### 知识回顾

- Module 类是 torch.nn 模块里提供的一个模型构造类 (nn.Module)，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型；

- PyTorch模型定义应包括两个主要部分：各个部分的初始化（_*init*_）；数据流向定义（forward）

  ![image-20220315145520403](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315145520403.png)

基于nn.Module，我们可以通过**Sequential，ModuleList和ModuleDict**三种方式定义PyTorch模型。

下面逐个探究这三种模型的定义方式

### Sequential

对应模块为nn.Sequential()。

当模型的前向计算为简单串联各个层的计算时， Sequential 类可以通过更加简单的方式定义模型。它可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 Module 的实例，⽽模型的前向计算就是将这些实例按添加的顺序逐⼀计算。我们结合Sequential和定义方式加以理解：

#### Sequential详解

```python
# Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))

```

参考：pytorch源码：https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential

**nn.Conv2d():**

参考：https://blog.csdn.net/qq_38863413/article/details/104108808

- ##### Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)

  in_channels：输入的通道数目 【必选】
  out_channels： 输出的通道数目 【必选】
  kernel_size：卷积核的大小，类型为int 或者元组，当卷积是方形的时候，只需要一个整数边长即可，卷积不是方形，要输入一个元组表示 高和宽。【必选】
  stride： 卷积每次滑动的步长为多少，默认是 1 【可选】
  padding： 设置在所有边界增加 值为 0 的边距的大小（也就是在feature map 外围增加几圈 0 ），例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5 。即在外围加了一圈 0 。【可选】
  dilation：控制卷积核之间的间距【可选】

  **dilation?**

  如果我们设置的dilation=0的话，效果如图：（蓝色为输入，绿色为输出，卷积核为3 × 3）

  ![image-20220315153901863](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315153901863.png)

  如果设置的是dilation=1，那么效果如图：（蓝色为输入，绿色为输出，卷积核仍为 3 × 3 。）
  但是这里卷积核点与输入之间距离为1的值相乘来得到输出。

  ![](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315153937602.png)

  groups：控制输入和输出之间的连接。（不常用）【可选】
  举例来说：
  比如 groups 为1，那么所有的输入都会连接到所有输出
  当 groups 为 2的时候，相当于将输入分为两组，并排放置两层，每层看到一半的输入通道并产生一半的输出通道，并且两者都是串联在一起的。这也是参数字面的意思：“组” 的含义。
  需要注意的是，in_channels 和 out_channels 必须都可以整除 groups，否则会报错（因为要分成这么多组啊，除不开你让人家程序怎么办？）

  bias： 是否将一个 学习到的 bias 增加输出中，默认是 True 。【可选】
  padding_mode ： 字符串类型，接收的字符串只有 “zeros” 和 “circular”。【可选】
  *注意：参数 kernel_size，stride，padding，dilation 都可以是一个整数或者是一个元组，一个值的情况将会同时作用于**高和宽** 两个维度，两个值的元组情况代表分别作用于 **高** 和 **宽** 维度。*

首先看初始化函数

![image-20220315151303536](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315151303536.png)

首先看一下初始函数`__init__`，在初始化函数中，首先是if条件判断，如果传入的参数为1个，并且类型为OrderedDict，通过字典索引的方式将子模块添加到`self._module`中，否则，通过for循环遍历参数，将所有的子模块添加到`self._module`中。**注意，Sequential模块的初始化函数没有异常处理，所以在写的时候要注意，注意，注意了**

接下来在看一下`forward`函数的实现：
因为每一个module都继承于nn.Module,都会实现`__call__`与`forward`函数，（具体见https://blog.csdn.net/dss_dssssd/article/details/82977170）所以forward函数中通过for循环依次调用添加到`self._module`中的子模块，最后输出经过所有神经网络层的结果。

![image-20220315152338945](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315152338945.png)

**三层网络结构的例子**

```python
# hyper parameters
in_dim=1
n_hidden_1=1
n_hidden_2=1
out_dim=1

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

      	self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True)，
            nn.Linear(n_hidden_1, n_hidden_2)，
            nn.ReLU(True)，
            # 最后一层不需要添加激活函数
            nn.Linear(n_hidden_2, out_dim)
             )

  	def forward(self, x):
      	x = self.layer(x)
      	return x

```

参考：https://github.com/swpucwf/pytorch_just_for_juan/blob/main/%E9%99%88%E4%BC%9F%E5%B3%B0/PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89/ch1.md

#### key-value形式（字典形式）

```python
import torch
from torch import nn
from collections import OrderedDict

class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, x):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            # self._modules.values 保存的是key-value
            print(module)
            x = module(x)
        return x
    
if __name__ == '__main__':
    args = OrderedDict([
                  ('conv1', nn.Conv2d(3,20,(5,5))),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,(5,5))),
                  ('relu2', nn.ReLU())
                ])
    model = MySequential(args)
    x = torch.randn(1,3,224,224)
    print(model)
    print(model(x).shape)
```

**输出**

```python
MySequential(
  (conv1): Conv2d(3, 20, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
)
Conv2d(3, 20, kernel_size=(5, 5), stride=(1, 1))
ReLU()
Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
ReLU()
torch.Size([1, 64, 216, 216])
```

#### 直接排列式

```python
import torch
from torch import nn
from collections import OrderedDict


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, x):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            # self._modules.values 保存的是key-value
            print(module)
            x = module(x)
        return x


if __name__ == '__main__':
    args = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model = MySequential(args)
    x = torch.randn(1,784)
    print(model)
    print(model(x).shape)

```

**输出**

```python
MySequential(
  (0): Sequential(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=10, bias=True)
  )
)
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
torch.Size([1, 10])
```



### ModuleList

对应模块为nn.ModuleList()。

ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。

**为什么要有ModuleList？**

写一个module然后就写foreword函数很麻烦，所以就有了这两个。它被设计用来**存储任意数量**的nn. module。

**什么时候用？**

如果在构造函数`__init__`中用到list、tuple、dict等对象时，一定要思考是否应该用ModuleList或ParameterList代替。

如果你想设计一个神经网络的层数作为输入传递。

**和list区别？**

ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。

当添加 nn.ModuleList 作为 nn.Module 对象的一个成员时（**即当我们添加模块到我们的网络时**），**所有 nn.ModuleList 内部的 nn.Module 的 parameter 也被添加作为 我们的网络的 parameter**。

要特别注意的是，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起。ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用for循环即可完成：

```python
class model(nn.Module):
  def __init__(self, ...):
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```

```python
import torch
from torch import nn


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.ModuleList = args

    def forward(self, x):
        for m in self.ModuleList:
            x = m(x)
        return x


if __name__ == '__main__':
    args = nn.ModuleList([
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 10)
    ])
    net = model(args)
    print(net)
    # net1(
    #   (modules): ModuleList(
    #     (0): Linear(in_features=10, out_features=10, bias=True)
    #     (1): Linear(in_features=10, out_features=10, bias=True)
    #   )
    # )

    for param in net.parameters():
        print(type(param.data), param.size())

    # class 'torch.Tensor'> torch.Size([256, 784])
    # class 'torch.Tensor'> torch.Size([256])
    # class 'torch.Tensor'> torch.Size([10, 256])
    # class 'torch.Tensor'> torch.Size([10])
    x = torch.randn(1,784)
    print(net)
    print(net(x).shape)
```

**输出**

```python
model(
  (ModuleList): ModuleList(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=10, bias=True)
  )
)
<class 'torch.Tensor'> torch.Size([256, 784])
<class 'torch.Tensor'> torch.Size([256])
<class 'torch.Tensor'> torch.Size([10, 256])
<class 'torch.Tensor'> torch.Size([10])
model(
  (ModuleList): ModuleList(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=10, bias=True)
  )
)
torch.Size([1, 10])
```

### ModuleDict

对应模块为nn.ModuleDict()。

ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称。

```python
import torch
from torch import nn


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.ModuleDict = args

    def forward(self, x):
        for key,value in self.ModuleDict.items():
            print("当前层：",key)
            print("当前输出",value(x).shape)
            x = value(x)
        # for m in self.ModuleDict.values():
        #     x = m(x)
        return x


if __name__ == '__main__':
    args = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU()
    })
    args['output'] = nn.Linear(256, 10)
    # 添加

    net = model(args)
    x = torch.randn(1,784)
    print(net(x).shape)
```

**输出**

```python
当前层： linear
当前输出 torch.Size([1, 256])
当前层： act
当前输出 torch.Size([1, 256])
当前层： output
当前输出 torch.Size([1, 10])
torch.Size([1, 10])
```

### 适用场景

Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写__init__和forward；

ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。



## 二：使用模型快速搭建复杂网络

### 学习目标

- 利用上一节的知识，将简单的层层构造成具有特定功能的模型块
- 使用模型构建复杂网络

### 模型搭建的基本方法

1. 模型块分析
2. 模型块实现
3. 利用模型块组装模型

### U-Net简介

UNet模型借以医学图像为代表学习的分割领域。U-Net 网络的模型结构，通过残差的连接结构解决了模型中的问题，是问题的广泛分布，使得神经网络的能力不断扩展。

![image-20220315160931427](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20220315160931427.png)

**模型块分析**

结合上图，不难发现U-Net模型具有非常好的对称性。模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接；同时位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”。此外还有输入和输出处理等其他组成部分。由于模型的形状非常像英文字母的“U”，因此被命名为“U-Net”。

**组成U-Net的模型块主要有如下几个部分：**

1）每个子块内部的两次卷积（Double Convolution）

2）左侧模型块之间的下采样连接，即最大池化（Max pooling）

3）右侧模型块之间的上采样连接（Up sampling）

4）输出层的处理

### U-Net模型块实现

在使用PyTorch实现U-Net模型时，我们不必把每一层按序排列显式写出，这样太麻烦且不宜读，一种比较好的方法是先定义好模型块，再定义模型块之间的连接顺序和计算方式。就好比装配零件一样，我们先装配好一些基础的部件，之后再用这些可以复用的部件得到整个装配体。

这里的基础部件对应上一节分析的四个模型块，根据功能我们将其命名为：DoubleConv, Down, Up, OutConv。下面给出U-Net中模型块的PyTorch 实现：

#### **引入**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

#### **每个子块内部的两次卷积（Double Convolution）**

```python
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

#### **左侧模型块之间的下采样连接`Down`，通过Max pooling来实现**

```python
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

```

#### **右侧模型块之间的上采样连接`Up`**

```python
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

```

#### **输出层的处理`OutConv`**

```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

#### **模型块之间的横向连接，输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现。**

### 利用模块组装U-Net

使用写好的模型块，可以非常方便地组装U-Net模型。可以看到，通过模型块的方式实现了代码复用，整个模型结构定义所需的代码总行数明显减少，代码可读性也得到了提升。

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

```python
import torch
from torch.nn import functional as F


class CNNLayer(torch.nn.Module):
    def __init__(self, C_in, C_out):
        '''
        卷积层
        :param C_in:
        :param C_out:
        '''
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(torch.nn.Module):
    def __init__(self, C):
        '''
        下采样
        :param C:
        '''
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C,kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):

        return self.layer(x)


class UpSampling(torch.nn.Module):

    def __init__(self, C):
        '''
        上采样
        :param C:
        '''
        super(UpSampling, self).__init__()
        self.C = torch.nn.Conv2d(C, C // 2, kernel_size=(1,1), stride=(1,1))

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.C(up)
        return torch.cat((x, r), 1)


class Unet(torch.nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        self.C1 = CNNLayer(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = CNNLayer(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = CNNLayer(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = CNNLayer(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = CNNLayer(512, 1024)
        self.U1 = UpSampling(1024)
        self.C6 = CNNLayer(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = CNNLayer(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = CNNLayer(256, 128)
        self.U4 = UpSampling(128)


        self.C9 = CNNLayer(128, 64)
        self.pre = torch.nn.Conv2d(64, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        U型结构
        :param x:
        :return:
        '''
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return self.sigmoid(self.pre(O4))


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256) #.cuda()
    net = Unet() #.cuda()
    print(net(a).shape)

```

```python
torch.Size([2, 3, 256, 256])
```

## 三：pytorch修改模型

除了自己构建PyTorch模型外，还有另一种应用场景：我们已经有一个现成的模型，但该模型中的部分结构不符合我们的要求，为了使用模型，我们需要对模型结构进行必要的修改。随着深度学习的发展和PyTorch越来越广泛的使用，有越来越多的开源模型可以供我们使用，很多时候我们也不必从头开始构建模型。因此，掌握如何修改PyTorch模型就显得尤为重要。

本节我们就来探索这一问题。经过本节的学习，你将收获：

- 如何在已有模型的基础上：
  - 修改模型若干层
  - 添加额外输入
  - 添加额外输出

### 修改模型层

我们这里以pytorch官方视觉库torchvision预定义好的模型ResNet50为例，探索如何修改模型的某一层或者某几层。我们先看看模型的定义是怎样的：

```python
import torchvision.models as models
net = models.resnet50()
print(net)

ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
..............
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

这里模型结构是为了适配ImageNet预训练的权重，因此最后全连接层（fc）的输出节点数是1000。

假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层，将其输出节点数替换为10。另外，我们觉得一层全连接层可能太少了，想再加一层。可以做如下修改：

```python
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
    
net.fc = classifier
```

这里的操作相当于将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构，该结构是我们自己定义的。这里使用了第一节介绍的Sequential+OrderedDict的模型定义方式。至此，我们就完成了模型的修改，现在的模型就可以去做10分类任务了。

