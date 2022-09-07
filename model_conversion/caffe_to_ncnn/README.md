##### 5.1 可选：在windows上把caffe模型转ncnn（生成一个文件）
+ ncnn是一个为手机端极致优化的高性能神经网络前向计算框架；
+ 在windows操作系统上，编辑caffe2ncnn.bat文件；
+ 设置参数：caffe的proto文件，caffe的参数文件，存储的ncnn格式文件名；
+ 直接运行该bat文件，即可得到ncnn格式的模型

##### 5.2 在linux上编译并安装ncnn
（1）拉取一个ubuntu18.04的基础镜像，构建容器作为运行环境：
```shell
$ apt-get update
$ apt-get install git vim g++ make cmake

# 安装protobuf，不然编译时不会生成caffe2ncnn这个文件
$ apt-get install libprotobuf-dev protobuf-compiler
# 查看protoc是否安装成功
$ protoc --version

# 找到libprotobuf所在的目录，例如我的是/usr/lib/x86_64-linux-gnu/
$ find / -name libprotobuf.so
# 设置C++环境变量，在~/.bashrc文件后添加下面这行
$ export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
# 让配置生效
$ source ~/.bashrc
```
（2）可选：下载并安装C++版本的opencv，使用源码安装
```shell
$ mkdir build
$ cd build 
$ cmake ..
$ make -j8
$ make install
```

（3）下载github上的ncnn项目，编译，测试是否安装成功
```shell
$ git clone https://github.com/Tencent/ncnn

# 注意：如果要使用gpu，需要下载安装vulkan
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j4
$ make install

# copy examples/squeezenet_v1.1.param to build/examples
# copy examples/squeezenet_v1.1.bin to build/examples
$ cd build/examples
$ ./squeezenet yourimage.jpg 

# output top-3 class-id and score
# you may refer examples/synset_words.txt to find the class name
# 404 = 0.990290
# 908 = 0.004464
# 405 = 0.003941
```

##### 5.3 使用tencent/ncnn项目将caffe模型转为ncnn格式
```shell
# 编译并安装完ncnn以后，将caffe模型转换为ncnn
$ cd /ncnn/tools/caffe/
$ ./caffe2ncnn pfld.prototxt pfld.caffemodel pfld.param pfld.bin
```

##### 5.4 下载并安装github上的pybind11项目
```shell
# 不安装pip和pytest可能会出错
$ apt-get update
$ apt-get install python3-pip 
$ pip3 install -i https://mirrors.aliyun.com/pypi/simple pytest

# 下载安装pybind11
$ git clone https://github.com/pybind/pybind11
$ cd <pyband11-root-dir>
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j4
$ make install

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:/<relative-root>/pybind11
```

##### 5.5 下载并安装github上的pyncnn项目
```shell
$ pip3 install -i https://mirrors.aliyun.com/pypi/simple numpy
$ git clone https://github.com/caishanli/pyncnn
$ cd /path/to/pyncnn
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/ncnn/build/install/lib/cmake/ncnn/ ..
$ make
$ cd ../python
$ pip3 install .

# 测试pyncnn
cd /path/to/pyncnn/tests
python3 test.py
```

##### 5.6 使用pyncnn测试ncnn模型
```shell
# 安装必要的包
$ apt-get install libsm6 libxrender1 libxext6 libglib2.0-dev
$ pip3 install -i https://mirrors.aliyun.com/pypi/simple tqdm requests portalocker opencv-python
# 先看看pyncnn/examples/squeezenet.py能不能跑通
$ python3 squeezenet.py example.jpg

# 如果自动下载squeezenet的参数文件失败，可以手动下载下面这个项目，把两个参数文件放到~/.ncnn/models目录
$ git clone https://github.com/caishanli/pyncnn-assets

# 更改/pyncnn/python/ncnn/model_zoo/squeezenet.py文件，设置为自己的模型处理过程
# 更改pyncnn/examples/squeezenet.py文件，第27行的net()改为自己的模型
# 注意ncnn模型和caffe模型相比，把输入层给去掉了；要改的测试代码部分较多
python3 test_ncnn.py example.jpg
```

