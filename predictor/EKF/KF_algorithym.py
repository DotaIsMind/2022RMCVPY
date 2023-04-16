# -*- coding:utf-8 -*-
# Data: 2021/12/16 下午9:03

import numpy as np
import matplotlib.pyplot as plt
from random import *


class KalmanFilter(object):
    def __init__(self):
        # ------------ PRED ----------------
        # 状态变量 X (4*1)   [yaw, pit, yaw_speed, pit_speed]
        self.Xk = np.zeros((4, 1), dtype=np.float32)
        # 状态转移矩阵A (4*4)
        # 状态转移  next_yaw = current_yaw + yaw_speed * delay_time
        # 状态转移矩阵，不需要控制矩阵  Xpred = A * Xt
        self.A = np.array([ [1, 0, 1, 0],
                            [0, 1, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)

        # 预测过程噪声的方差   P = A * P * A^T + R
        self.R = np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32) * 0.1
        # ----------------- UPDATE ----------------
        # 观测矩阵
        self.H = np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        # self.H = np.array([ [1, 0, 0, 0],
        #                     [0, 1, 0, 0]],dtype=np.float32)

        # 测量噪声的方差
        self.Q = np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32) * 20
        # self.Q = np.array([[1, 0],
        #                    [0, 1]], dtype=np.float32) * 0.01

        # 定义卡尔曼增益 K = P * H^T * (H * P * H^T + Q)^-1
        self.K = np.zeros((4, 4), dtype=np.float32)
        # self.K = np.zeros((4, 2), dtype=np.float32)

        # 定义估计误差方差
        self.P = np.eye(4, dtype=np.float32)

        self.last_measurement = np.zeros((4, 1), dtype=np.float32)
        self.error = 0

    def track(self, x, y, x_speed, y_speed, delay_time):
        # 初始化状态变量Xk
        self.Xk = np.array([x,
                            y,
                            x_speed,
                            y_speed], dtype=np.float32).reshape(4,1)
        # ------------ PRED -------------
        #  Xpred = A * Xt
        # 设置转移矩阵中的时间项目，乘以状态变量中的速度为预测位移
        self.A[0, 2] = delay_time * 1
        self.A[1, 3] = delay_time * 1
        # self.A[1, 3] = 1
        # self.A[1, 3] = 1

        # if self.last_measurement is not None:
        #     bool_target_change = self.check_target_change()
        #     if bool_target_change:
        #         self.KF_reset()

        Xpred = np.dot(self.A, self.Xk)
        # 预测值和真实值之间的误差协方差矩阵 P(n|n-1)=A * P(n-1|n-1) * A^T + R
        self.P = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.R
        # ------------update ------------
        # 计算卡尔曼增益 Kg(k)= P(k|k-1) H^T / (H P(k|k-1) H^T + Q)
        self.K = np.dot(np.dot(self.P, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.transpose()) + self.Q))
        # print(self.K)

        # 修正结果，计算滤波值
        self.Xk = Xpred + np.dot(self.K, (self.Xk - np.dot(self.H, Xpred)))

        # 计算估计值和真实值之间误差协方差矩阵
        self.P = np.dot((np.eye(4) - np.dot(self.K, self.H)), self.P)
        self.last_measurement = Xpred

        return self.Xk[0][0], self.Xk[1][0], self.Xk[2][0], self.Xk[3][0]
        #return x, y, 0, 0

    def KF_reset(self):
        # 预测过程噪声的方差   P = A * P * A^T + R
        self.R = np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32) * 1

        # 测量噪声的方差
        self.Q = np.array([ [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32) * 10
        # self.Q = np.array([[1, 0],
        #                    [0, 1]], dtype=np.float32) * 0.01

        # 定义卡尔曼增益 K = P * H^T * (H * P * H^T + Q)^-1
        self.K = np.zeros((4, 4), dtype=np.float32)
        # self.K = np.zeros((4, 2), dtype=np.float32)

        # 定义估计误差方差
        self.P = np.eye(4, dtype=np.float32)

        self.last_measurement = None

    def check_target_change(self):
        # 切换目标误差
        error = self.Xk - np.dot(self.H, self.last_measurement)
        # 残差分布
        D_k = np.dot(np.dot(self.H,  self.P), self.H.transpose()) + self.R
        # 残差值
        R_k = np.dot(np.dot(error.transpose(), np.linalg.inv(D_k)),  error)
        print(self.Xk, R_k)
        if R_k < 5:
            return False
        else:
            return True


class TwoDimKF(object):
    def __init__(self):
        # ------------ PRED ----------------
        # 状态变量 X (4*1)   [yaw, pit, yaw_speed, pit_speed]
        self.Xk = np.zeros((2, 1), dtype=np.float32)
        # 状态转移矩阵A (4*4)
        # 状态转移  next_yaw = current_yaw + yaw_speed * delay_time
        # 状态转移矩阵，不需要控制矩阵  Xpred = A * Xt
        self.A = np.array([ [1, 1],
                            [0, 1],], dtype=np.float32)

        # 预测过程噪声的方差   P = A * P * A^T + R
        self.R = np.array([ [1, 0],
                            [0, 1]], dtype=np.float32) * 1
        # ----------------- UPDATE ----------------
        # 观测矩阵
        self.H = np.array([ [1, 0],
                            [0, 1]], dtype=np.float32)
        # self.H = np.array([ [1, 0, 0, 0],
        #                     [0, 1, 0, 0]],dtype=np.float32)

        # 测量噪声的方差
        self.Q = np.array([ [1, 0],
                            [0, 1]], dtype=np.float32) * 1
        # self.Q = np.array([[1, 0],
        #                    [0, 1]], dtype=np.float32) * 0.01

        # 定义卡尔曼增益 K = P * H^T * (H * P * H^T + Q)^-1
        self.K = np.zeros((2, 2), dtype=np.float32)
        # self.K = np.zeros((4, 2), dtype=np.float32)

        # 定义估计误差方差
        self.P = np.eye(2, dtype=np.float32)

        self.last_measurement = np.zeros((2, 1), dtype=np.float32)
        self.error = 0

    def track(self, x, x_speed, delay_time):
        # 初始化状态变量Xk
        self.Xk = np.array([[x],
                            [x_speed]], dtype=np.float32)
        # ------------ PRED -------------
        #  Xpred = A * Xt
        # 设置转移矩阵中的时间项目，乘以状态变量中的速度为预测位移
        self.A[0, 1] = delay_time * 1

        # if self.last_measurement is not None:
        #     bool_target_change = self.check_target_change()
        #     if bool_target_change:
        #         self.KF_reset()

        Xpred = np.dot(self.A, self.Xk)
        # 预测值和真实值之间的误差协方差矩阵 P(n|n-1)=A * P(n-1|n-1) * A^T + R
        self.P = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.R
        # ------------update ------------
        # 计算卡尔曼增益 Kg(k)= P(k|k-1) H^T / (H P(k|k-1) H^T + Q)
        self.K = np.dot(np.dot(self.P, self.H.transpose()), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.transpose()) + self.Q))
        # print(self.K)

        # 修正结果，计算滤波值
        self.Xk = Xpred + np.dot(self.K, (self.Xk - np.dot(self.H, Xpred)))

        # 计算估计值和真实值之间误差协方差矩阵
        self.P = np.dot((np.eye(2) - np.dot(self.K, self.H)), self.P)
        self.last_measurement = Xpred

        return self.Xk[0][0], self.Xk[1][0]
        # return x, x_speed

    def reset(self):
        # 预测过程噪声的方差   P = A * P * A^T + R
        # 预测过程噪声的方差   P = A * P * A^T + R
        self.R = np.array([ [1, 0],
                            [0, 1]], dtype=np.float32) * 0.1

        # 测量噪声的方差
        self.Q = np.array([ [1, 0],
                            [0, 1]], dtype=np.float32) * 10
        # self.Q = np.array([[1, 0],
        #                    [0, 1]], dtype=np.float32) * 0.01

        # 定义卡尔曼增益 K = P * H^T * (H * P * H^T + Q)^-1
        self.K = np.zeros((2, 2), dtype=np.float32)
        # self.K = np.zeros((4, 2), dtype=np.float32)

        # 定义估计误差方差
        self.P = np.eye(2, dtype=np.float32)

        self.last_measurement = None

    def check_target_change(self):
        # 切换目标误差
        error = self.Xk - np.dot(self.H, self.last_measurement)
        # 残差分布
        D_k = np.dot(np.dot(self.H,  self.P), self.H.transpose()) + self.R
        # 残差值
        R_k = np.dot(np.dot(error.transpose(), np.linalg.inv(D_k)),  error)
        print(self.Xk, R_k)
        if R_k < 5:
            return False
        else:
            return True


if __name__ == '__main__':
    pitch = np.arange(1, 10 * np.pi, 0.3)
    yaw = np.sin(pitch)
    # pitch = yaw * 2
    yaw_speed = [yaw[i] - yaw[i-1] for i in range(1, len(yaw))]
    yaw_speed.append(yaw_speed[len(yaw_speed)-1])
    # last_pit = pitch

    plt.figure(1)

    plt.subplot(221)
    plt.plot(pitch, yaw, color='r', linestyle='-', marker='o')
    plt.title("KFtrack")
    plt.xlabel("x")
    plt.ylabel("yaw")

    p_x = []
    p_y = []
    error_x = []
    error_y = []

    # # ----- 4 dim KF -----
    # a = KalmanFilter()
    #
    # for i in range(len(pitch)):
    #     p_yaw, p_pit, yaw_s, pit_s = a.track(yaw[i], pitch[i], yaw_speed[i], 0.3, 15)
    #     # print(p_pit, p_yaw)
    #     # print("x:")
    #     # print(int(nx))
    #     # print("y:")
    #     # print(int(ny))
    #     p_x.append(p_pit)
    #     p_y.append(p_yaw)
    #     error_y.append(p_x[i] - pitch[i])
    #     error_x.append(p_y[i] - yaw[i])
    #
    # # plt.subplot(221)
    # plt.plot(p_x, p_y, color='g', linestyle='--', marker='o')
    # plt.legend(["y=sin(x)", "KFtrack"], loc='best')
    # plt.show()
    #
    # plt.subplot(223)
    # plt.plot(error_x, linestyle=":")
    # plt.xlabel('yaw')
    # plt.ylabel('yaw_error')
    #
    # plt.subplot(222)
    # plt.plot(error_y, linestyle=":")
    # plt.xlabel('pit')
    # plt.ylabel('pit_error')
    # plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    # plt.show()
    # # -----end ----

    # yaw_angle_f = open("2022-04-30-11_49_41yaw.txt")
    # yaw_angle = yaw_angle_f.readlines()
    # frame_list = []
    # yaw_list = []
    # pre_list = []
    # for i in range(0, len(yaw_angle), 1):
    #     yaw_split = yaw_angle[i].split(" ")
    #     frame_list.append(np.int(yaw_split[0]))
    #     yaw_list.append(np.float(yaw_split[1]))
    #     pre_list.append(np.float(yaw_split[2]))


    b = TwoDimKF()
    b.R = b.R * 0.1
    b.Q = b.Q * 10
    for i in range(len(yaw)):
        p_yaw, p_yaw_s = b.track(yaw[i], yaw_speed[i], 20)
        p_x.append(p_yaw)
        p_y.append(p_yaw_s)
        error_x.append(p_x[i] - yaw[i])
        error_y.append(p_y[i] - yaw_speed[i])

    plt.subplot(221)
    plt.plot(pitch, p_x, color='g', linestyle='--', marker='o')
    plt.legend(["y=sin(x)", "KFtrack"], loc='best')

    plt.subplot(223)
    plt.plot(pitch, error_x, linestyle="--", marker='o')
    plt.xlabel('yaw')
    plt.ylabel('yaw_error')

    plt.subplot(222)
    plt.plot(pitch, error_y, linestyle="--", marker='o')
    plt.xlabel('yaw_s')
    plt.ylabel('yaw_s_error')
    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.show()














