# pytorch进阶训练

参考：https://github.com/datawhalechina/thorough-pytorch/blob/main/%E7%AC%AC%E5%85%AD%E7%AB%A0%20PyTorch%E8%BF%9B%E9%98%B6%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/6.1%20%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md

https://github.com/swpucwf/pytorch_just_for_juan/blob/main/%E9%99%88%E4%BC%9F%E5%B3%B0/%E8%BF%9B%E9%98%B6%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/ch2.md

## 6.1 自定义损失函数

PyTorch在torch.nn模块为我们提供了常用的MSELoss，L1Loss，BCELoss……但是随着深入学习的发展，越来越多的非官方提供的损失函数，例如，这些 DiceLos，HuberLoss，SobolevLos……损失函数是针对一些非通用的我们需要的模型，PyTorch 不能全部添加到库中去，因此这些损失函数通过自定义函数来实现另外，在科学研究中，我们会提出全新的损失功能来提升模型的表现，表现我们无法使用 PyTorch 自带的功能，也没有相关的博客供，此时参考实现损失功能就可以了那件事更为重要了。

本节的学习，你将收获：

- 如何掌握自定义损失函数

### 6.1.1 以函数方式定义

事实上，损失函数仅仅是一个函数而已，因此我们可以通过直接以函数定义的方式定义一个自己的函数，如下所示：

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```

```python
import numpy as np
import torch

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss



if __name__ == '__main__':
    target = torch.randn(20)
    output = torch.randn(20)
    print(my_loss(target,output))
```

```python
tensor(1.9438)
```

### 6.1.2 以类方式定义

以函数定义的很简单，但是以类定义的方式常用，以发现类的方式定义损失函数的时候，我们如果可以看每一个损失函数的继承关系，我们就可以`Loss`函数部分继承`_loss`，部分继承自`_WeightedLoss`，而`_WeightedLoss`继承自`_loss`，` _loss`继承自**nn.Mod**。我们可以将我们作为其神经网络的等级来，同样地，的损失类就需要继承自**nn.Module**，在下面的例子中我们以Dice Loss类为例向大家讲述。

Dice Loss 是一种在分割领域中常见的损失函数，定义如下：

dice loss参考https://zhuanlan.zhihu.com/p/86704421



$$ DSC = \frac{2|X∩Y|}{|X|+|Y|} $$ 实现代码如下：

```python
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
	def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# 使用方法    
criterion = DiceLoss()
loss = criterion(input,targets)
```

```python
import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


if __name__ == '__main__':
    inputs = torch.ones(224,224)
    targets = torch.zeros(224,224)
    # 使用方法
    criterion = DiceLoss()
    loss = criterion(inputs, targets)
    print(loss)
```

```python
tensor(1.0000)
```

比如说，U经常的损失函数还有BCE-Dice Loss，Jaccard/section over Union (Io) Loss，Focal Loss......

```python
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                     
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
--------------------------------------------------------------------
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
--------------------------------------------------------------------
    
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
# 更多的可以参考链接1
```

**注：**

在损失函数的时候，涉及到全局自定义的时候，我们最好用Torch提供计算接口，这样就不需要实现自动请求引导功能，并且我们可以直接调用cuda，使用numpy或者scipy的数学关于PyTorch使用类定义伤害函数的原因，可以PyTorch的讨论区（链接6）

### 本节参考

【1】https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/notebook
【2】https://www.zhihu.com/question/66988664/answer/247952270
【3】https://blog.csdn.net/dss_dssssd/article/details/84103834
【4】[https://zj-image-processing.readthedocs.io/zh_CN/latest/pytorch/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0/](https://zj-image-processing.readthedocs.io/zh_CN/latest/pytorch/自定义损失函数/)
【5】https://blog.csdn.net/qq_27825451/article/details/95165265
【6】https://discuss.pytorch.org/t/should-i-define-my-custom-loss-function-as-a-class/89468

## 6.2 动态调整学习率

学习率的选择是深度学习中一个困扰人们许久的问题，学习速率设置过小，会极大降低收敛速度，增加训练时间；学习率太大，可能导致参数在最优解两侧来回振荡。但是当我们选定了一个合适的学习率后，**经过许多轮的训练后，可能会出现准确率震荡或loss不再下降等情况，说明当前学习率已不能满足模型调优的需求。此时我们就可以通过一个适当的学习率衰减策略来改善这种现象，提高我们的精度**。这种设置方式在PyTorch中被称为scheduler，也是我们本节所研究的对象。

经过本节的学习，你将收获：

- 如何根据需要选取已有的学习率调整策略
- 如何自定义设置学习调整策略并实现

### 6.2.1 使用官方scheduler

- **了解官方提供的API**

在训练神经网络的过程中，学习率是最重要的超参数之一，作为当前较为流行的深度学习框架，PyTorch已经在`torch.optim.lr_scheduler`为我们封装好了一些动态调整学习率的方法供我们使用，如下面列出的这些scheduler。

- [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
- [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
- [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
- [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
- [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)
- [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
- [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
- [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
- [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
- [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

- **使用官方API**

关于如何使用这些动态调整学习率的策略，`PyTorch`官方也很人性化的给出了使用实例代码帮助大家理解，我们也将结合官方给出的代码来进行解释。

```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
	scheduler1.step() 
	...
    schedulern.step()
```

**注**：

我们在使用官方给出的`torch.optim.lr_scheduler`时，需要将`scheduler.step()`放在`optimizer.step()`后面进行使用。

### 6.2.2 自定义scheduler

虽然PyTorch官方给我们提供了许多的API，但是在实验中也有可能碰到需要我们自己定义学习率调整策略的情况，而我们的方法是自定义函数`adjust_learning_rate`来改变`param_group`中`lr`的值，在下面的叙述中会给出一个简单的实现。

假设我们现在正在做实验，需要学习率每30轮下降为原来的1/10，假设已有的官方API中没有符合我们需求的，那就需要自定义函数来实现学习率的改变。

```python
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

有了`adjust_learning_rate`函数的定义，在训练的过程就可以调用我们的函数来实现学习率的动态变化

```python
def adjust_learning_rate(optimizer,...):
    ...
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
```

### 本节参考

【1】[PyTorch官方文档](https://pytorch.org/docs/stable/optim.html)

## 6.3 模型微调

随着深度学习的发展，模型的参数越来越大，许多开源模型都是在较大数据集上进行训练的，比如Imagenet-1k，Imagenet-11k，甚至是ImageNet-21k等。但在实际应用中，我们的数据集可能只有几千张，这时从头开始训练具有几千万参数的大型神经网络是不现实的，因为越大的模型对数据量的要求越大，过拟合无法避免。

假设我们想从图像中识别出不同种类的椅⼦，然后将购买链接推荐给用户。一种可能的方法是先找出100种常见的椅子，为每种椅子拍摄1000张不同⻆度的图像，然后在收集到的图像数据集上训练一个分类模型。这个椅子数据集虽然可能比Fashion-MNIST数据集要庞⼤，但样本数仍然不及ImageNet数据集中样本数的十分之⼀。这可能会**导致适用于ImageNet数据集的复杂模型在这个椅⼦数据集上过拟合。同时，因为数据量有限，最终训练得到的模型的精度也可能达不到实用的要求。**

为了应对上述问题，一个显⽽易⻅的解决办法是**收集更多的数据**。然而，收集和标注数据会花费大量的时间和资⾦。例如，为了收集ImageNet数据集，研究人员花费了数百万美元的研究经费。虽然目前的数据采集成本已降低了不少，但其成本仍然不可忽略。

另外一种解决办法是**应用迁移学习(transfer learning)，将从源数据集学到的知识迁移到目标数据集上**。例如，虽然ImageNet数据集的图像大多跟椅子无关，但在该数据集上训练的模型可以抽取较通用的图像特征，从而能够帮助识别边缘、纹理、形状和物体组成等。这些类似的特征对于识别椅子也可能同样有效。

**迁移学习的一大应用场景是模型微调（finetune）。简单来说，就是我们先找到一个同类的别人训练好的模型，把别人现成的训练好了的模型拿过来，换成自己的数据，通过训练调整一下参数。** 在PyTorch中提供了许多预训练好的网络模型（VGG，ResNet系列，mobilenet系列......），这些模型都是PyTorch官方在相应的大型数据集训练好的。学习如何进行模型微调，可以方便我们快速使用预训练模型完成自己的任务。

经过本节的学习，你将收获：

- 掌握模型微调的流程
- 了解PyTorch提供的常用model
- 掌握如何指定训练模型的部分层

### 6.3.1 模型微调的流程

1. 在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即**源模型**。

2. 创建一个新的神经网络模型，即**目标模型**。它复制了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。

3. **为目标模型添加一个输出⼤小为⽬标数据集类别个数的输出层，并随机初始化该层的模型参数**。

4. **在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。**

   

### 6.3.2 使用已有模型结构

这里我们以torchvision中的常见模型为例，列出了如何在图像分类任务中使用PyTorch提供的常见模型结构和参数。对于其他任务和网络结构，使用方式是类似的：

- 实例化网络

  ```python
  import torchvision.models as models
  resnet18 = models.resnet18()
  # resnet18 = models.resnet18(pretrained=False)  等价于与上面的表达式
  alexnet = models.alexnet()
  vgg16 = models.vgg16()
  squeezenet = models.squeezenet1_0()
  densenet = models.densenet161()
  inception = models.inception_v3()
  googlenet = models.googlenet()
  shufflenet = models.shufflenet_v2_x1_0()
  mobilenet_v2 = models.mobilenet_v2()
  mobilenet_v3_large = models.mobilenet_v3_large()
  mobilenet_v3_small = models.mobilenet_v3_small()
  resnext50_32x4d = models.resnext50_32x4d()
  wide_resnet50_2 = models.wide_resnet50_2()
  mnasnet = models.mnasnet1_0()
  ```

- 传递`pretrained`参数

通过`True`或者`False`来决定是否使用预训练好的权重，在默认状态下`pretrained = False`，意味着我们不使用预训练得到的权重，当`pretrained = True`，意味着我们将使用在一些数据集上预训练得到的权重。

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
```

**注意事项：**

1. 通常PyTorch模型的扩展为`.pt`或`.pth`，程序运行时会首先检查默认路径中是否有已经下载的模型权重，一旦权重被下载，下次加载就不需要下载了。

2. 一般情况下预训练模型的下载会比较慢，我们可以直接通过迅雷或者其他方式去 [这里](https://github.com/pytorch/vision/tree/master/torchvision/models) 查看自己的模型里面`model_urls`，然后手动下载，预训练模型的权重在`Linux`和`Mac`的默认下载路径是用户根目录下的`.cache`文件夹。在`Windows`下就是`C:\Users\<username>\.cache\torch\hub\checkpoint`。我们可以通过使用 [`torch.utils.model_zoo.load_url()`](https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url)设置权重的下载地址。

3. 如果觉得麻烦，还可以将自己的权重下载下来放到同文件夹下，然后再将参数加载网络。

   ```python
   self.model = models.resnet50(pretrained=False)
   self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
   ```

4. 如果中途强行停止下载的话，一定要去对应路径下将权重文件删除干净，要不然可能会报错。

### 6.3.3 训练特定层

在默认情况下，参数的属性`.requires_grad = True`，如果我们从头开始训练或微调不需要注意这里。但如果我们**正在提取特征并且只想为新初始化的层计算梯度，其他参数不进行改变**。那我们就需要通过设置`requires_grad = False`来**冻结部分层**。在PyTorch官方中提供了这样一个例程。

```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```

在下面我们仍旧使用`resnet18`为例的将1000类改为4类，但是仅改变最后一层的模型参数，**不改变特征提取的模型参数**；注意我们**先冻结模型参数的梯度，再对模型输出部分的全连接层进行修改，这样修改后的全连接层的参数就是可计算梯度的。**

```python
import torchvision.models as models
# 冻结参数的梯度
feature_extract = True
model = models.resnet18(pretrained=True)
set_parameter_requires_grad(model, feature_extract)
# 修改模型
num_ftrs = model.fc.in_features
model.fc = nn.Linear(in_features=512, out_features=4, bias=True)
```

之后在训练过程中，**model仍会进行梯度回传，但是参数更新则只会发生在fc层**。通过设定参数的requires_grad属性，我们完成了指定训练模型的特定层的目标，这对实现模型微调非常重要。

### 本节参考

【1】[参数更新](https://www.pytorchtutorial.com/docs/package_references/torch-optim/)
【2】[给不同层分配不同的学习率](https://blog.csdn.net/jdzwanghao/article/details/90402577)

## 6.4 半精度训练

我们提到PyTorch时候，总会想到要用硬件设备GPU的支持，也就是“卡”。**GPU的性能主要分为两部分：算力和显存，前者决定了显卡计算的速度，后者则决定了显卡可以同时放入多少数据用于计算**。在可以使用的显存数量一定的情况下，每次训练能够加载的数据更多（也就是batch size更大），则也可以提高训练效率。另外，有时候数据本身也比较大（比如3D图像、视频等），显存较小的情况下可能甚至batch size为1的情况都无法实现。因此，合理使用显存也就显得十分重要。

我们观察PyTorch默认的浮点数存储方式用的是torch.float32，小数点后位数更多固然能保证数据的精确性，但绝大多数场景其实并不需要这么精确，只保留一半的信息也不会影响结果，也就是使用torch.float16格式。由于数位减了一半，因此被称为“半精度”。

显然半精度能够减少显存占用，使得显卡可以同时加载更多数据进行计算。本节会介绍如何在PyTorch中设置使用半精度计算。

经过本节的学习，你将收获：

- 如何在PyTorch中设置半精度训练
- 使用半精度训练的注意事项

### 6.4.1 半精度训练的设置

在PyTorch中使用autocast配置半精度训练，同时需要在下面三处加以设置：

- **import autocast**

```python
from torch.cuda.amp import autocast
```

- **模型设置**

在模型定义中，使用python的装饰器方法**，用autocast装饰模型中的forward函数**。关于装饰器的使用，可以参考[这里](https://www.cnblogs.com/jfdwd/p/11253925.html)：

```python
@autocast()   
def forward(self, x):
    ...
    return x
```

- **训练过程**

在训练过程中，只需在**将数据输入模型及其之后的部分放入“with autocast()**:“即可：

```python
 for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)
        ...
```

**注意：**

**半精度训练主要适用于数据本身的size比较大**（比如说3D图像、视频等）。当数据本身的size并不大时（比如手写数字MNIST数据集的图片尺寸只有28*28），使用半精度训练则可能不会带来显著的提升。

**注意：**

1、网络要在GPU上跑，模型和输入样本数据都要cuda().half()

2、模型参数转换为half型，不必索引到每层，直接model.cuda().half()即可

3、对于半精度模型，优化算法，Adam我在使用过程中，在某些参数的梯度为0的时候，更新权重后，梯度为零的权重变成了NAN，这非常奇怪，但是Adam算法对于全精度数据类型却没有这个问题。

　　另外，SGD算法对于半精度和全精度计算均没有问题。

 