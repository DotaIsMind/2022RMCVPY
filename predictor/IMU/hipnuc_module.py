#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' For hipnuc module '

import serial
import json
from queue import Queue
from ModuleTest.IMUTest.hipnuc_protocol import *
import time

# 用于测试

class hipnuc_module(object):
    """超核模组

    超核模组类，用于接收，处理超核模组的信息

    Parameters
    ----------
    path_configjson : str
        json配置文件的路径.

    """
    def __init__(self, path_configjson=None):

        def serialthread():
            while self.serthread_alive:
                # 如果串口有数据，则接收至缓冲区
                if self.serial.in_waiting:
                    # 读取串口
                    data = self.serial.read(self.serial.in_waiting)
                    # 放至缓冲区
                    self.binbuffer.extend(data)
                else:
                    pass

                # 解析缓冲区数据
                try:
                    while True:
                        #尝试查找完整帧,若失败会抛出异常
                        headerpos, endpos = intercept_one_complete_frame(self.binbuffer)
                        #解析完整帧
                        extraction_information_from_frame(self.binbuffer[headerpos :endpos + 1],self.module_data_fifo,self.config["report_datatype"])
                        self.binbuffer = self.binbuffer[endpos + 1:]

                except HipnucFrame_NotCompleted_Exception as NotCompleted:
                    #接收进行中
                    pass
                except HipnucFrame_ErrorFrame_Exception as e:
                    print(e)
                    #目前帧有帧头，但是为错误帧，跳过错误帧
                    headerpos = find_frameheader(self.binbuffer)
                    self.binbuffer = self.binbuffer[headerpos + 1:]
                # finally:
                #     pass

                time.sleep(0.010)

        # 解析json配置文件
        if path_configjson != None:
            # 打开配置文件
            config_json = open(path_configjson, 'r', encoding='utf-8')
            self.config = json.load(config_json)
            # 关闭配置文件
            config_json.close()
            # 进行配置
            portx = self.config["port"]
            bps = self.config["baudrate"]
        else:
            pass

        # 初始化串口
        # 打开串口，并得到串口对象
        self.serial = serial.Serial(portx, bps, timeout=None)
        # FIFO
        self.module_data_fifo = Queue()

        self.binbuffer = []

        self.serthread_alive = True
        self.serthread = threading.Thread(target=serialthread)
        self.serthread.start()

        self.sample_timer = None
        self.sample_timer = threading.Timer(1.00, sample_rate_timer_cb, args=(self.sample_timer,))
        self.sample_timer.start()

    def get_module_data(self,timeout = None):
        """获取数据.

        获取已接收到的模组数据.

        Parameters
        ----------
        timeout :
            可选参数。若为None(默认值),将会阻塞直至有有效值;
            若timeout为正数，将会尝试等待有效数据并阻塞timeout秒,若阻塞时间到仍未有有效数据,将会抛出Empty异常.

        Returns
        -------
        data : dict
            返回模组数据，类型为字典

        """


        data = self.module_data_fifo.get(block=True,timeout=timeout)
        return data

    def get_module_data_size(self):
        """获取数据数量.

        获取已接收到的模组数据的数量.
        注意:返回长度大于0,不保证get_module_data时不会被阻塞.

        Parameters
        ----------
        无

        Returns
        -------
        size : int
            返回模组数据，类型为字典

        """

        return self.module_data_fifo.qsize()

    def close(self):
        """关闭模组.

        关闭指定的模组.

        Parameters
        ----------
        无

        Returns
        -------
        无

        """
        self.serthread_alive = False
        sample_rate_timer_close()
        self.serial.close()