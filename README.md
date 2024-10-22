# ModelQuantization
介绍在不同框架下对ResNet18的量化
# TODO
+ [] pytorch
+ [] onnx
+ [] tensorrt
+ [] ncnn
# 系统配置
Windows 11、cpu Intel Core i5-11400F
# 权重来源
[resnet18](https://download.pytorch.org/models/resnet18-f37072fd.pth)

# 不同框架下的性能损耗
## 静态量化
|框架|计算精度|推理速度|推理准确性|
|:---:|:---:|:---:|:---:|
|pytorch|float32|34.7405ms|0.76|
|pytorch|int8|10.5147ms|0.73|
## 动态量化
# pytorch
##  环境介绍
+ pytorch 2.4.1+cu118
## 文件介绍
+ resnet: resnet18的模型，应用静态量化训练和动态量化训练，两种方式进行量化: 1.静态量化训练(PTSQ)，2.动态量化训练
## 运行方法
进入pytorch目录
```python main.py```
# ONNX
## 环境介绍
onnxruntime
## 文件介绍
+ main.py: 导出onnx
+ run.py: onnx量化
