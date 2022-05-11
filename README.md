# `Icraft`版`ByteTrack`

## 一、背景

#### 问题1：

截止到`V1.1`版本，`Icraft`暂时只原生支持`ReLU`和`LeakyReLU`两种激活函数，而`ByteTrack`使用了`SiLU`激活函数

可行的解决方法有以下几种：

1. 等待`IcraftV2.0`上线。该版本新增了对若干激活函数的原生支持，包括`SiLU`
2. 使用`CustomOp`软算子。`Icraft`支持自定义软算子，但由于使用`CPU`计算，且激活函数出现频繁，导致数据传输时间过长。

4. 修改模型，使用`LeakyReLU`替换`SiLU`作为激活函数，需要重新训练

本工程提供方案三的参考代码

#### 问题2：

`ByteTrack`的`backbone`使用切片式的`Focus`作为输入层

截止到`V1.1`版本，`Icraft`暂时不支持`Python`中的切片操作

本工程将其改为了卷积层

#### 问题3：

`ByteTrack`未提供模型导出代码

本工程提供了模型导出功能，以便`Icraft`编译

导出模型时删除掉最后一层卷积后的操作，它们将在后处理代码中实现

#### 问题4：

截止到`V1.1`版本，`Icraft`的阈值筛选加速硬算子要求阈值必须在每个检测层的第`1`个特征图内

本工程导出模型时调整计算顺序，先计算含`obj_output`的卷积层

模型导出和`Icraft`加载模型，均以计算发生的顺序为准



## 二、使用步骤

### 2.1、准备工作

下载[MOT17](https://motchallenge.net/data/MOT17/)数据集，将其放到`datasets`目录下

使用以下脚本生成`MOT17`的标签

```powershell
python tools/convert_mot17_to_coco.py # 耗时很久
```

在`Windows`下直接安装`cython_bbox`会失败，可以用以下方法替换：

```powershell
python -m pip install git+https://github.com/yanfengliu/cython_bbox.git
```
安装`ByteTrack`
```powershell
python setup.py develop
```

下载权重`bytetrack_s_mot17.pth.tar`并放到`pretrained`目录下

### 2.2、训练

训练开始前，可以在`exps\example\mot\yolox_s_mix_det.py`中修改模型信息

使用以下代码即可开始训练

```bash
python tools/train.py -f exps/example/mot/yolox_s_mix_det.py -d 1 -b 12 --fp16 -o -c pretrained/bytetrack_s_mot17.pth.tar
```

使用以下代码即可恢复最近一次训练

```bash
python tools/train.py -f exps/example/mot/yolox_s_mix_det.py -d 1 -b 12 --fp16 -o -c pretrained/bytetrack_s_mot17.pth.tar --resume
```

### 2.3、确认推理正确

使用以下代码即可推理

```bash
python tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/latest_ckpt.pth.tar --fp16 --fuse --save_result
```

### 2.4、测试精度

使用以下代码即可测试精度

```powershell
python tools/track.py -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/latest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse
```

> label好像还没处理好，正在修复

### 2.5、保存模型

使用以下代码即可保存模型

```bash
python tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/latest_ckpt.pth.tar --fp16 --fuse --save_result --export True
```



## 三、修改部分

`setup.py`：指定`utf-8`编码读取

`tool\demo_track.py`：增加导出模型参数

`yolox\models\darknet.py`：将`Focus`替换为卷积

`yolox\models\network_blocks.py`：将`SiLU`替换为`LeakyReLU`

`yolox\models\yolo_head.py`：将`obj_output`计算顺序提至最前；删除掉卷积后的操作

