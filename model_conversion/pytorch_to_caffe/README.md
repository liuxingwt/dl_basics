#### 3. Pytorch模型转caffe
##### 3.1 失败的github项目1：
+ https://github.com/xxradon/PytorchToCaffe
+ 环境：Pytorch1.3  
+ 跑demo`python3 example/alexnet_pytorch_to_caffe.py`时，  
+ 报错：TypeError: None has type NoneType, but expected one of: bytes, unicode  
+ 这个issue作者至今未解决：https://github.com/xxradon/PytorchToCaffe/issues/68 本次尝试失败

##### 3.2 成功的github项目2：
+ https://github.com/hahnyuan/nn_tools
+ 环境：Pytorch1.0
+ 同样是报上面的错误，查找issue后重新搭建一个pytorch1.0的环境，跑demo终于没问题了；
+ caffe模型的可视化工具网站：http://ethereon.github.io/netscope/#/editor
```shell
# pytorch模型转换为caffe：
python3 pfld_pytorch_to_caffe.py
# caffe模型解析parse
python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,3,224,224

# 测试caffe模型是否能跑通，首先通过docker镜像安装caffe的cpu版本，测试caffe是否安装成功
import caffe
print(caffe.__version__)

# 运行test_caffe.py文件，查看landmark结果是否正确
# 需要注意测试代码中有三处net.blobs['blob0']的key需要改成caffe模型输入的key
python test_caffe.py

# caffe模型在cpu环境里运行成功。但在gpu版本里运行会报以下错误：
# 这是因为官方提供的镜像bvlc/caffe：gpu最高只支持到cuda8.0 GTX960的GPU？？？（此处不确定）如果要运行得自行编译caffe；
`cudnn_conv_layer.cpp:53] Check failed: status == CUDNN_STATUS_SUCCESS (4 vs. 0)  CUDNN_STATUS_INTERNAL_ERROR`
```

##### 3.3 使用caffe的C++接口进行推理
+ 参考资料：caffe项目的示例——使用C++写的分类器：
+ https://github.com/BVLC/caffe/tree/master/examples/cpp_classification
+ 

##### 把人脸防伪的resnet50转换为caffe模型
+ （1）在1.4机器的pytorch容器mas-dl-torch2caffe文件夹中把pytroch模型转化为caffe，max—pool层可能会有问题找康乐的环境转；
+ （1.1）转完以后可以在1.4的caffe容器中，test_caffe文件夹下测试哪一层出了问题
+ （2）转化完的prototxt文件，需要把60行左右的max-pool层，加上offset
+ （3）转化完的prototxt文件，需要把最后面的avgpool层，kernel_h和kernel_w重新设置；
+ （4）转化完的prototxt文件，需要在最后加上一个softmax层；
