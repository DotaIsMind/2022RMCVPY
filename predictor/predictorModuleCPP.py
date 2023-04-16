# -*- coding:utf-8 -*-
# Data: 2021/11/20 上午3:39

# -*- coding:utf-8 -*-
# Data: 2021/11/19 下午11:59
import traceback

from common.msgQueue import Q_detector2predictor
from predictor.kalmanfilter.TrackKF_2D import KalmanFilter
from common.logFile import log_file_path
# ---- KF publisher ----
import Message_A
from Message_A import A

import time
import sys
import cv2
import numpy as np


kfpub = Message_A.Publisher("pub_KF")
def pub_anglemsg2cpp(pre_x, pre_y):
    global kfpub
    t = time.time()
    a = A()
    a.str = "hello, KF"
    a.pre_x = np.float32(pre_x)
    a.pre_y = np.float32(pre_y)

    while True:
        if time.time() - t < 0.5:
            pass
        else:
            kfpub.push(a)
            t = time.time()

# global current_measurement, last_measurement, current_prediction, last_prediction

# last_measurement = current_measurement = np.array((2, 1), np.float32)
#
# last_prediction = current_prediction = np.zeros((2, 1), np.float32)
# kalman = cv2.KalmanFilter(4, 2)  # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
# kalman.measurementMatrix = np.array([[1, 0, 0, 0],
#                                      [0, 1, 0, 0]], np.float32)  # 系统测量矩阵
# kalman.transitionMatrix = np.array([[1, 0, 1, 0],
#                                     [0, 1, 0, 1],
#                                     [0, 0, 1, 0],
#                                     [0, 0, 0, 1]], np.float32)  # 状态转移矩阵
# kalman.processNoiseCov = np.array([[1, 0, 0, 0],
#                                    [0, 1, 0, 0],
#                                    [0, 0, 1, 0],
#                                    [0, 0, 0, 1]],
#                                   np.float32) * 0.03  # 系统过程噪声协方差
KF_pre = KalmanFilter()
def KFpredictor_main():
    try:
        frame_id = 0
        last_measurement = np.zeros((1, 2))
        while True:
            if not Q_detector2predictor.empty():
                current_measurement = Q_detector2predictor.get()
                pre_x, pre_y = KF_pre.track(current_measurement[0][0], current_measurement[0][1])
                last_measurement = current_measurement
                pub_anglemsg2cpp(float(1), float(2))
            else:
                pre_x, pre_y = KF_pre.track(last_measurement[0][0], last_measurement[0][1])
                print("Q_detector2predictor is empty, use last measurement!")
            # 每20ms发送一次
            time.sleep(0.02)
            frame_id += 1
            print("predict frame id:", frame_id)
            #todo: predict result trans to electric controll
    except:
        # traceback.print_exc(log_file_path, "a")
        traceback.print_exc()


# if __name__ == "__main__":
#     pub_anglemsg2cpp(float(1), float(2))

