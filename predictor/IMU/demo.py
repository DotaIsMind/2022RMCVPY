#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' a test module '

from ModuleTest.IMUTest.hipnuc_module import *
import time
import serial
import serial.tools.list_ports

def open_ports():
    # 查找串口
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) <= 0:
        print("The serial ports can not be find!")
    else:
        print("Find {} ports!".format(len(port_list)))
        print(port_list)
        return port_list[0]


if __name__ == '__main__':

    HI221GW = hipnuc_module('./config.json')
    # uart_port = open_ports()
    # ser = serial.Serial(uart_port.device, 115200, timeout=1)
    while True:
        data = HI221GW.get_module_data()

        print(data)

        time.sleep(3)