# -- coding:utf-8 --
# Time:2022/1/8 19:22
# Author:
import time

import matplotlib.pyplot as plt
import numpy as np
from ctypes import *
import struct
from predictor.EKF.KF_algorithym import TwoDimKF
import serial
from common.logFile import LOGGING


def pack_COM_data(id, delay_time, yaw_angle, pit_angle):
    # class cv2COM(Structure):
    #     _fields_ = [("flag_id", c_uint8),
    #                   ("id", c_uint32),
    #                   ("yaw_target_angle", c_float),
    #                   ("pit_target_angle", c_float),
    #                   ("flag_t", c_uint8)]
    # 固定值
    flag_id_t = c_uint8(0xAA).value
    flag_t_t = c_uint8(0x55).value
    # todo: 做参数类型检查
    id_t = c_uint32(id).value
    # 秒转化为毫秒
    delay_time_t = (c_float(delay_time).value)    # delay_time_t = c_float(0.030).value

    yaw_angle_t = (c_float(yaw_angle).value)
    pit_angle_t = (c_float(pit_angle).value)
    cv2COM_info = [flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, flag_t_t]
    # uint8-usigned_char: B, uint32-usigned_int: I, float-float: f
    pack_fmt = "@BIfffB"
    COM_strm = struct.pack(pack_fmt, flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, flag_t_t)
    # COM_obj = cv2COM(*cv2COM_info)
    return COM_strm


def read_data_vis(file_path):
    yaw_angle_f = open(file_path, 'r')
    yaw_angle = yaw_angle_f.readlines()
    frame_list = []
    yaw_list = []
    pre_list = []
    yaw_s_list = []
    error_list = []
    for i in range(0, len(yaw_angle), 1):
        yaw_split = yaw_angle[i].split(" ")
        frame = np.int(yaw_split[0])
        yaw = np.float(yaw_split[1])
        pre_yaw = np.float(yaw_split[2])
        yaw_v = np.float(yaw_split[3])
        frame_list.append(frame)
        yaw_list.append(yaw)
        pre_list.append(pre_yaw)
        yaw_s_list.append(yaw_v)
        error_list.append(pre_yaw-yaw)

    plt.title("KFtrack")
    plt.xlabel("time")
    plt.ylabel("angle")

    plt.plot(frame_list, yaw_list, color="r", linestyle="-", marker="o")
    plt.plot(frame_list, pre_list, color="g", linestyle="--", marker="o")
    plt.plot(frame_list, error_list, color="b", linestyle="-", marker="o")

    plt.legend(["yaw", "pre_yaw", "error"])
    plt.show()


def read_data_pre(file_path):
    yaw_angle_f = open(file_path, 'r')
    yaw_angle = yaw_angle_f.readlines()
    frame_list = []
    yaw_list = []
    pre_list = []
    yaw_v_list = []
    error_list = []
    for i in range(0, len(yaw_angle), 1):
        yaw_split = yaw_angle[i].split(" ")
        frame = np.int(yaw_split[0])
        yaw = np.float(yaw_split[1])
        pre_yaw = np.float(yaw_split[2])
        yaw_v = np.float(yaw_split[3])
        frame_list.append(frame)
        yaw_list.append(yaw)
        pre_list.append(pre_yaw)
        yaw_v_list.append(yaw_v)
        error_list.append(pre_yaw-yaw)

    b = TwoDimKF()
    b.R = b.R * 1
    b.Q = b.Q * 5
    p_x = []
    p_y = []
    error_x = []
    error_y = []
    for i in range(len(yaw_list)):
        p_yaw, p_yaw_s = b.track(yaw_list[i], yaw_v_list[i], 35)
        p_x.append(p_yaw)
        p_y.append(p_yaw_s)
        error_x.append(p_x[i] - yaw_list[i])
        error_y.append(p_y[i] - yaw_v_list[i])

    plt.title("KFtrack")
    plt.xlabel("time")
    plt.ylabel("angle")

    plt.plot(frame_list, yaw_list, color="r", linestyle="-", marker="o")
    plt.plot(frame_list, pre_list, color="g", linestyle="--", marker="o")

    plt.legend(["yaw", "pre_yaw", "error"])
    plt.show()

    plt.title("KFtrack")
    plt.xlabel("time")
    plt.ylabel("yaw error")
    plt.plot(frame_list, error_list, color="b", linestyle="-", marker="o")
    plt.show()


if __name__ == "__main__":
    # yaw_angle_path = "2022-04-30-11_59_32yaw.txt"
    yaw_angle_path = "2022-04-30-12_32_28yaw.txt"
    read_data_vis(yaw_angle_path)
    # read_data_pre(yaw_angle_path)

