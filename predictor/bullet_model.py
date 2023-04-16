# -- coding:utf-8 --
# Time:2022/3/16 16:39
# Author:
import time

import numpy as np
import math
import struct

from common.Config import ser_obj, CVMSG2COM
from common.logFile import LOGGING
import serial
import serial.tools.list_ports

# 空气阻力系数
# guard factor
# k = np.float32(0.020)
# spin infantry factor
k = np.float32(0.050)
# 重力系数
G = 9.78
# bullet speed
bullet_v = 20
shoot_delay = 0.090


def bullet_model(x, v, angle):
    '''
    @brief: 计算相机坐标y轴补偿
    @param: 相机坐标系x， 弹丸初速度， pit轴角度
    @return: y轴补偿值
    '''
    t = np.true_divide( np.exp(k * x) - 1, k * v * np.cos(angle))
    y = v * np.sin(angle) * t - G * t * t * 0.5
    return y


def get_angle_offset(x, y, dis):
    '''
    @brief: 计算添加弹道补偿后的pit轴角度
    @brief: 相机坐标系x, y坐标，弹丸初速度
    @return: pit轴补偿值
    '''
    pit = 0
    yaw_angle_v = 0
    # port_list = list(serial.tools.list_ports.comports())
    # if len(port_list) == 0:
    #     LOGGING.warning("No useful COM!")
    # ser_obj = serial.Serial(port_list[0].device, 115200)
    # if CVMSG2COM and ser_obj is not None:
    #     bullet_v_list = [15, 18, 19]
    #     fmt = "<BBfBB"
    #     read_size= struct.calcsize(fmt)
    #     r_data = ser_obj.read(read_size)
    #     if r_data != b'':
    #         r_data = struct.unpack(fmt, r_data)
    #         if r_data[0] == 0xAA and r_data[-1] == 0x55:
    #             angle_v = r_data[2]
    #             if abs(angle_v) < 20:
    #                 yaw_angle_v = angle_v
    #             b_v = r_data[3]
    #             if b_v in bullet_v_list:
    #                 bullet_v = b_v
    #         else:
    #             LOGGING.error("PRE:COM data FLAG data error")
    # print("angle_v", yaw_angle_v, "b_v", bullet_v)
    y_t = y
    for i in range(20):
        pit = np.arctan2(y_t, x)
        y_ac = bullet_model(x, bullet_v, pit)
        dy = y - y_ac
        # print("pit", pit * 180 / np.pi, "y_t", y_t, "y_ac", y_ac, "dy", dy)
        y_t = y_t + dy
        if (np.abs(dy)< 0.001):
            break

    # # fly_t = dis / bullet_v # m/s
    # yaw_offset = np.arctan2(0.12 * bullet_v, dis) * 180 / np.pi
    # LOGGING.info("PRE:yaw offset-{}".format(yaw_offset))
    # # todo:解决快速通信问题
    # # yaw_offset = 0
    return pit


if __name__ == "__main__":
    x = 4
    y = 3
    dis = 5
    while True:
        t1 = time.time()
        pit, yaw_offset = get_angle_offset(x, y, dis)
        print(pit * 180 / np.pi, yaw_offset)
        print((time.time() - t1) * 1000)
        # time.sleep(0.020)
    # if CVMSG2COM:
    # if True:
    #     port_list = list(serial.tools.list_ports.comports())
    #     if len(port_list) == 0:
    #         LOGGING.warning("No useful COM!")
    #     ser_obj = serial.Serial(port_list[0].device, 115200)
    #
    # while True:
    #     fmt = "<bbfbb"
    #     read_size= struct.calcsize(fmt)
    #     r_data = ser_obj.read(read_size)
    #     # r_data = "AA0333338BC10F55"
    #     # r_data = bytes.fromhex(r_data)
    #     print(r_data.hex())
    #     # r_data = struct.unpack(fmt, r_data)
    #     if r_data != b'':
    #         # r_data = struct.unpack(fmt, r_data)
    #         if r_data[0] == 0xAA and r_data[-1] == 0x55:
    #             angle_v = struct.unpack("<f", r_data[2:6])
    #             bullet_v = r_data[6]
    #             print(r_data[7])
    #
    #             print(r_data)
    #         else:
    #             LOGGING.error("PRE:COM data FLAG data error")
    #     else:
    #         print("have no COM data")
    #         time.sleep(1)
