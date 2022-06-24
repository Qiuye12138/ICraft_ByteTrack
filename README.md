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

安装`cython_bbox`会失败，可以用以下方法替换：

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

                IDF1   IDP   IDR  Rcll  Prcn   GT  MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm num_objects
MOT17-02-DPM   54.5% 66.0% 46.5% 61.0% 86.6%   53  18  24  11  931  3849  82   163 50.8% 0.216  64  21  11        9880
MOT17-04-DPM   84.0% 84.8% 83.2% 93.9% 95.7%   69  62   6   1 1008  1477  62    87 89.5% 0.142  27  24   6       24178
MOT17-04-SDP   84.0% 84.8% 83.2% 93.9% 95.7%   69  62   6   1 1008  1477  62    87 89.5% 0.142  27  24   6       24178
MOT17-04-FRCNN 84.0% 84.8% 83.2% 93.9% 95.7%   69  62   6   1 1008  1477  62    87 89.5% 0.142  27  24   6       24178
MOT17-11-DPM   71.0% 75.7% 66.9% 83.0% 93.8%   44  20  19   5  248   769  21    38 77.0% 0.143  17   9   5        4517
MOT17-10-FRCNN 68.5% 74.9% 63.0% 78.5% 93.4%   36  17  17   2  330  1272  47    92 72.2% 0.220  29  20   8        5923
MOT17-11-SDP   71.0% 75.7% 66.9% 83.0% 93.8%   44  20  19   5  248   769  21    38 77.0% 0.143  17   9   5        4517
MOT17-13-DPM   71.6% 79.8% 64.9% 75.7% 93.1%   44  24  12   8  176   766  25    39 69.4% 0.215  20   9   5        3156
MOT17-10-SDP   68.5% 74.9% 63.0% 78.5% 93.4%   36  17  17   2  330  1272  47    92 72.2% 0.220  29  20   8        5923
MOT17-09-DPM   71.8% 77.2% 67.1% 81.8% 94.2%   22  14   8   0  144   523  20    35 76.1% 0.172  20   4   6        2879
MOT17-05-DPM   66.6% 72.4% 61.6% 80.2% 94.2%   71  34  30   7  167   666  29    50 74.3% 0.174  35  12  21        3357
MOT17-09-FRCNN 71.8% 77.2% 67.1% 81.8% 94.2%   22  14   8   0  144   523  20    35 76.1% 0.172  20   4   6        2879
MOT17-10-DPM   68.5% 74.9% 63.0% 78.5% 93.4%   36  17  17   2  330  1272  47    92 72.2% 0.220  29  20   8        5923
MOT17-05-FRCNN 73.5% 80.3% 67.8% 79.7% 94.4%   71  32  32   7  159   681  30    49 74.1% 0.174  33  13  17        3357
MOT17-02-FRCNN 54.5% 66.0% 46.5% 61.0% 86.6%   53  18  24  11  931  3849  82   163 50.8% 0.216  64  21  11        9880
MOT17-02-SDP   54.5% 66.0% 46.5% 61.0% 86.6%   53  18  24  11  931  3849  82   163 50.8% 0.216  64  21  11        9880
MOT17-09-SDP   71.8% 77.2% 67.1% 81.8% 94.2%   22  14   8   0  144   523  20    35 76.1% 0.172  20   4   6        2879
MOT17-13-FRCNN 71.6% 79.8% 64.9% 75.7% 93.1%   44  24  12   8  176   766  25    39 69.4% 0.215  20   9   5        3156
MOT17-05-SDP   66.6% 72.4% 61.6% 80.2% 94.2%   71  34  30   7  167   666  29    50 74.3% 0.174  35  12  21        3357
MOT17-13-SDP   71.6% 79.8% 64.9% 75.7% 93.1%   44  24  12   8  176   766  25    39 69.4% 0.215  20   9   5        3156
MOT17-11-FRCNN 71.0% 75.7% 66.9% 83.0% 93.8%   44  20  19   5  248   769  21    38 77.0% 0.143  17   9   5        4517
OVERALL        74.1% 79.0% 69.7% 82.7% 93.7% 1017 565 350 102 9004 27981 859  1511 76.6% 0.168 634 298 182      161670
```


### 2.5、保存模型

使用以下代码即可保存模型

```bash
python tools/demo_track.py video -f exps/example/mot/yolox_s_mix_det.py -c YOLOX_outputs/yolox_s_mix_det/latest_ckpt.pth.tar --save_result --export True
```



## 三、修改部分

`setup.py`：指定`utf-8`编码读取

`tool\demo_track.py`：增加导出模型参数

`yolox\models\darknet.py`：将`Focus`替换为卷积

`yolox\models\network_blocks.py`：将`SiLU`替换为`LeakyReLU`

`yolox\models\yolo_head.py`：将`obj_output`计算顺序提至最前；删除掉卷积后的操作

