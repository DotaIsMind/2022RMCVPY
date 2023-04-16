# -*- coding:utf-8 -*-
# Data: 2022/3/20 下午3:21

import struct
import time
from ctypes import *
import serial
import serial.tools.list_ports
import numpy as np
import random

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
    dis = (c_float(0).value)
    cv2COM_info = [flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, dis, flag_t_t]
    # uint8-usigned_char: B, uint32-usigned_int: I, float-float: f
    pack_fmt = "@BIffffB"
    COM_strm = struct.pack(pack_fmt, flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, dis, flag_t_t)
    # COM_obj = cv2COM(*cv2COM_info)
    return COM_strm


def square_wave():
    yaw_angle = 4
    pit_angle = 0
    send_cnt = 1
    port_list = list(serial.tools.list_ports.comports())
    print(port_list)
    if len(port_list) == 0:
        print('无可用串口')
    ser = serial.Serial(port_list[0].device, 115200)
    while True:
        if send_cnt % 5 == 0:
            yaw_angle = yaw_angle * -1
            pit_angle = pit_angle * -1
        msg = pack_COM_data(send_cnt, 16, yaw_angle, pit_angle)

        rst = ser.write(msg)
        if rst:
            print("Send msg len {}".format(len(msg)))
        send_cnt += 1

        time.sleep(2)


def sin_angle():
    p_x = []
    p_y = []
    error_x = []
    error_y = []
    pitch = np.arange(1, 10 * np.pi, 0.3)
    yaw = 3 * np.sin(pitch)
    # pitch = yaw * 2
    yaw_speed = [yaw[i] - yaw[i-1] for i in range(1, len(yaw))]
    yaw_speed.append(yaw_speed[len(yaw_speed)-1])
    #kf_obj = KalmanFilter()

    port_list = list(serial.tools.list_ports.comports())
    print(port_list)
    if len(port_list) == 0:
        print('无可用串口')
    send_cnt = 1
    ser = serial.Serial(port_list[0].device, 115200, timeout=1)
    while True:
        for i in range(0, len(yaw)):
            #p_yaw, p_pit, p_yaw_s, p_pit_s = kf_obj.track(yaw[i], pitch[i], yaw_speed[i], 0.3, 40)

            msg = pack_COM_data(send_cnt, 40, yaw[i], 0)

            rst = ser.write(msg)
            if rst:
                print("Send msg len {}, yaw: {}, pit: {}".format(len(msg), yaw, pitch))
            send_cnt += 1

            time.sleep(0.020)


if __name__ == "__main__":
    square_wave()
    # sin_angle()

