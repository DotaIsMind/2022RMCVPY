#NVIDIA Jetson NX部署YOLOX笔记

##1.利用官方工具刷系统镜像
[官方刷镜像教程](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#write):https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#write)
:https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#write
，官方预装了Ubuntu 18.04，和NVIDIA官方包，详细的信息在：(https://developer.nvidia.cn/zh-cn/embedded/jetpack)
此次安装为Jet pack 4.6版本。

###(1)按照常规Ubuntu系统设置时间语言等进入桌面后，利用`sudo spt-get update`更新软件依赖，
然后使用`sudo apt-get install python3-pip`，使用`pip -V`查看更新后的版本。

###(2)添加CUDA的环境变量
使用`vi ~/.bashrc`打开文件，在文件末尾添加<br>
`export PATH=/usr/local/cuda/bin:$PATH`<br>
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`<br>
使用`source ~/.bashrc`使配置生效，使用`nvcc -V`查看CUDA版本。

###(3)！！！预警提示！！！
- 官方镜像使用python=3.6.9为自带的python版本，在使用Tensorrt时如果使用高版本的python，Tensorrt会
提示python版本不匹配。
- 最好是使用python自带的虚拟环境部署环境，如果要使用**anaconda、miniconda**等第三方工具，因为Jetson NX是ARM64架构
的系统，anaconda和miniconda最低支持版本都是python=3.7，并且无法通过`conda create -n env_name python=3.6`创建
版本的python环境，即使更换国内源也不行，此文是通过Miniforge[github地址Miniforge-pypy3-Linux-aarch64](https://github.com/conda-forge/miniforge)
实现的python=3.6的虚拟环境的创建，然后使用`sudo find / -iname "*tensorrt*"`查找到Tensorrt的路径，<br>显示
`/usr/lib/python3.6/dist-packages/tensorrt`
- 然后进入
miniforge创建的虚拟环境路径`miniforge3/envs/env_name/lib/python3.6/site-packages`下通过软链接链接接Tensorrt的路径
`ln -s /usr/lib/python3.6/dist-packages/tensorrt/tensorrt.so tensorrt.so`，如果链接.so文件失败可尝试直接链接
tensorrt文件。
- 另外注意在安装anaconda、miniconda、miniforge时不建议使用`sudo`命令，普通用户会产生文件读写权限不够安装失败的问题。

###(4)安装onnx、onnxruntime、onnxruntime-gpu、torch
以上安装包都可以在Jetson Zoo网站(https://www.elinux.org/Jetson_Zoo) 找到，注意和python版本的对应关系，
下载完成以后使用`pip3 install `安装
**wheel**文件即可。
- 注意Jetson Zoo下载的为onnxruntime-gpu版本，没有gpu后缀的为CPU版本，只安装onnxruntime可能造成运行YOLOX官方Demo
时推理速度太慢。

###(5)安装torchvision
torch和torchvision都需要提前安装依赖，具体命令可以查看torch官方Doc，
torch版本要和torchvision版本对应，对应关系可以在官方git仓库查到，地址:(https://github.com/pytorch/vision) ，点击
`main`按钮下在`Tags`中选择历史版本，下在完成后按照源码编译即可。

###(6)安装torch2trt导出trt模型
下载官方源码编译安装即可，网址：(https://github.com/NVIDIA-AI-IOT/torch2trt)

###(7)安装大恒相机驱动
2022赛季的视觉硬件选择了大恒USB3.0相机，在官网下载[地址](https://www.daheng-imaging.com/index.php?m=content&c=index&a=lists&catid=59&czxt=&sylx=21&syxj=#mmd)安装了官方python驱动以后，会出现找不到动态链接库的问题，下载C++版本的
驱动重新编译安装即可，同时会在源码文件夹下生成Glaxyview软件，可使用该软件检测相机是否正常工作。

###(8)cmake、gcc、g++升级
官方自带镜像gcc和g++为7.0版本，在部署YOLOX过程中未遇到需要升级的情况。

###(9)设置开启启动脚本
- NX风扇自启，教程(https://cloud.tencent.com/developer/article/1647455) ，也可以使用
`sudo -H pip install -U jetson-stats`直接安装，然后运行`jtop`，进入Ctrl页面修改开机启动即可。使用Jtop是设置最大
转速，如果要自定义转速参照教程: https://blog.csdn.net/u013963960/article/details/107360244?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_antiscanv2&utm_relevant_index=1

- 设置开机启动脚本时在启动语句后面加符号`&`设置为后台启动，防止脚本内含有死循环卡在命令界面无法进入图形界面，如果不慎写入了
死循环语句，可以把NX的内存卡取出，放到另一台Linux设备上打开，修改/etc/rc.local文件后再放到NX上重新启动。

