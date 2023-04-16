# -*- coding:utf-8 -*-
# Data: 2021/11/19 下午11:59

import traceback

from common.msgQueue import Q_detector2predictor
from predictor.EKF.KF_algorithym import KalmanFilter, TwoDimKF
from predictor.bullet_model import *
from common.logFile import LOGGING, log_file_path, time_fmt
from common.Config import WRITE_VIDEO
from common import Config
from camera.camera_calibrate.camera_calib_params import CameraCalibParamDic

import serial
import binascii
import struct
import time
from ctypes import *
import numpy as np
import cv2


log_cnt = 20
PRE_DEBUG_INFO = False
# bx_KF_pre = KalmanFilter()
# ---- Init Two Dim Kalman Filter ----
yaw_KF = TwoDimKF()
# 预测噪声R较小，更相信预测值
yaw_KF.R = yaw_KF.R * 0.1
yaw_KF.Q = yaw_KF.Q * 15
pit_KF = TwoDimKF()
# 预测噪声R较小，更相信预测值
pit_KF.R = pit_KF.R * 0.1
pit_KF.Q = pit_KF.Q * 2


def point_sort(box):
    '''
    @brief: 由二点坐标生成四点
    @param: armor box二点坐标
    @return: armor四个顶点
    '''
    # # x = [box[0][0], box[1][0], box[2][0], box[3][0]]
    # index = np.argsort(box)
    # left = [box[index[0]], box[index[1]]]
    # right = [box[index[2]], box[index[3]]]
    # if left[0] < left[1]:
    #     left_up = left[0]
    #     left_down = left[1]
    # else:
    #     left_up = left[1]
    #     left_down = left[0]
    # if right[0] < right[1]:
    #     right_up = right[0]
    #     right_down = right[1]
    # else:
    #     right_up = right[1]
    #     right_down = right[0]

    # YOLOX 输出对角点坐标
    left_up = [box[0], box[1]]
    left_down = [box[0], box[3]]
    right_up = [box[2], box[1]]
    right_down = [box[2], box[3]]
    return left_up, right_up, right_down, left_down


def pos_dis(point_pos1, point_pos2):
    return np.sqrt((point_pos1[0] - point_pos2[0]) ** 2 + (point_pos1[1] - point_pos2[1]) ** 2)


def box_area(box):
    '''
    @param: box left up and right down position
    @return 对角线长度，中心坐标"
    '''
    # 四点模型计算方法
    left_up, right_up, right_down, left_down = point_sort(box)

    # 利用海伦公式求解四边形最大面积
    # 有上-左上， 绝对值
    bx_a = pos_dis(right_up, left_up)
    # 右下-右上
    bx_b = pos_dis(right_down, right_up)
    # 左下-右下
    bx_c = pos_dis(left_down, right_down)
    # 左下-左上
    bx_d = pos_dis(left_down, left_up)
    # 对角线：右下-左上
    bx_diagonal = pos_dis(right_down, left_up)
    # 系数q=周长/2
    q1 = (bx_a + bx_b + bx_diagonal)/2
    q2 = (bx_c + bx_d + bx_diagonal)/2
    # 四边形面积=两个三角形相加
    bx_area = np.sqrt(q1/2 * (q1-bx_a)*(q1-bx_b)*(q1-bx_diagonal)) + \
              np.sqrt(q2/2 * (q2 - bx_c) * (q2 - bx_d) * (q2 - bx_diagonal))

    # # YOLOX对角线计算方法
    # left_up_x = box[0]
    # left_up_y = box[1]
    # right_down_x = box[2]
    # right_down_y = box[3]
    # # 对角线长度
    # bx_area = np.sqrt( (right_down_x - left_up_x) **2, (right_down_y - left_up_y) **2 )
    return bx_area


def box_cent(left_up, left_down, right_up):
    # left_up, left_down, right_up, right_down = point_sort(box)
    #装甲板中心 return a tuple
    bx_cent = [left_up[0] + (right_up[0] - left_up[0])/2, left_up[1] + (left_down[1]-left_up[1])/2]
    return bx_cent


def car_box_cent(left_down, right_down):
    # left_up, left_down, right_up, right_down = point_sort(box)
    #装甲板中心 return a tuple
    # bx_cent = [left_down[0] + (right_down[0] - left_down[0])/2, left_down[1] + (left_down[1]-left_up[1])/2]
    # todo: 丢失装甲板以后pitch计算角度由车的中心改为车y轴坐标的0.95, y轴向下
    bx_cent = [int(left_down[0] + (right_down[0] - left_down[0])/2), int(left_down[1] * 0.6)]
    return bx_cent


def gimbal_pit_yaw(left_up, right_up, right_down, left_down, bx_cent):
    #   原理是：：：PNP算法
    #   找到四个对应点，根据摄像头参数求解实际世界坐标
    #   找外接矩形的四个图像点
    #   分别设置为（0，0，0），（0，车体长度，0），（0，车体长度，车体高度），（0，0，车体高度）///
    #   但是这样做不对，因为车体在旋转过程中无法在图像上找到精确的位置，无法计算。
    #   应该以检测装甲板的位置作为四个对应点，这样他的大小是固定的“

    # 图像输入的点顺序与世界坐标系的点匹配左上，右上，右下，左下，单位：mm
    image_points = np.array([left_up, right_up, right_down, left_down, bx_cent], dtype="double")
    # todo:这个数值需要调优，因为根据YOLO检测的框面积比实际的装甲板大
    #high = 120 #mm
    #width = 130 #mm
    # big armor
    high = 130 #mm
    width = 230 #mm

    # 世界坐标系：以装甲板中心为原点,x轴向右，y轴向下的坐标系, 分别为左上，右上，右下，左下，单位：mm
    model_points = np.array([
        (-width/2, -high/2, 0),
        (width/2, -high/2, 0),
        (width/2, high/2, 0),
        (-width/2, high/2, 0),
        (0, 0, 0)])

    cam_params = CameraCalibParamDic[Config.CAR_ID]
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, 
                                                                  image_points, cam_params.infantry4_instrimat, cam_params.infantry4_dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    # solvePnP得到是旋转向量，使用罗格里德转化为旋转矩阵
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    # print("rotation vector norm: ", np.linalg.norm(rotation_vector))
    # 偏移矩阵就是目标在相机坐标系下的坐标
    # yaw_c: atan(x/z)  pit_c: atan(y/sqrt(x^2+z^2))
    yaw_c = np.arctan2(translation_vector[0], translation_vector[2]) * 180 / np.pi # 化为角度，陀螺仪角速度为角度/秒
    ballistic_x = np.sqrt(translation_vector[0] ** 2 + translation_vector[2] ** 2)
    # 坐标系y轴朝下，Pitch朝上旋转为正

    distance = np.linalg.norm(translation_vector) / 1000  # 单位:m
    pit = np.arctan2(-translation_vector[1], ballistic_x) *  180 / np.pi

    # if not pit_c[0]:
    # pit_c = np.arctan2(translation_vector[1], np.linalg.norm(translation_vector)) * 180 / np.pi * 0.1

    # 弹道补偿, 单位:m, 输入y坐标为向上正，向下负
    # yaw轴飞行时间补偿
    pit_c = get_angle_offset(ballistic_x / 1000, -translation_vector[1] / 1000, distance)
    pit_c = pit_c * 180 / np.pi
    # yaw_c += yaw_offset
    # print("Translation yaw:{}, pit:{}, dis: {}".format(yaw_c, pit_c, distance))

    # print("After ballistic compensation pit:{}, delt pit: {}".format(pit_c, np.round(pit_c-pit, 4)))
    # 这里得到的是相机在装甲板世界坐标系的位姿
    camera_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)

    return rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c[0], pit_c[0], distance


def big_small_end_convert(data):
    return binascii.hexlify(binascii.unhexlify(data)[::-1])


def floatToBytes(f):
    bs = struct.pack("f", f)
    return bs


def bytesToFloat(f):
    f1 = f[0]
    f2 = f[1]
    f3 = f[2]
    f4 = f[3]
    ba = bytearray()
    ba.append(f1)
    ba.append(f2)
    ba.append(f3)
    ba.append(f4)
    return struct.unpack("!f", ba)[0]


def pack_COM_data(id, delay_time, yaw_angle, pit_angle, distance):
    '''
    @brief: 通过struct模块打包字节数据
    @param: z帧ID， 延迟，角度
    @return: 数据字节流
    '''
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

    # pit_angle_t = (c_float(0).value)
    # cv2COM_info = [flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, flag_t_t]
    # uint8-usigned_char: B, uint32-usigned_int: I, float-float: f
    # ------ GUARD ----
    if Config.CAR_ID == 7 or Config.CAR_ID == 107:
        yaw_angle_t = (c_float(yaw_angle).value)
        pit_angle_t = (c_float(pit_angle).value)
        dis_t = (c_float(distance).value)
        #pack_fmt = "<BIffffB"
        pack_fmt = "@BIffffB"
        COM_strm = struct.pack(pack_fmt, flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, dis_t, flag_t_t)

    # ----- SPIN INFANTRY ----
    if Config.CAR_ID == 4 or Config.CAR_ID == 104:
        # yaw_angle_t = (c_float(yaw_angle).value)
        # pit_angle_t = (c_float(pit_angle).value)
        # dis_t = (c_float(distance).value)
        yaw_angle_t = (c_float(yaw_angle).value)
        pit_angle_t = (c_float(pit_angle).value)
        # dis_t = (c_float(3).value)
        pack_fmt = "@BffB"
        COM_strm = struct.pack(pack_fmt, flag_id_t, yaw_angle_t, pit_angle_t, flag_t_t)
        # print(COM_strm.hex())
    return COM_strm


def get_IOU(left_up, right_down, grey_left_up, grey_right_down):
    '''
    @brief:  由两个armor的二点坐标计算IOU
    @param: 两个armor的左上和右下坐标
    @return: IOU
    '''
    # boxA = [int(x) for x in boxA]
    # boxB = [int(x) for x in boxB]

    # xA = max(boxA[0], boxB[0])
    xA = max(left_up[0], grey_left_up[0])
    yA = max(left_up[1], grey_left_up[1])
    xB = min(right_down[0], grey_right_down[0])
    yB = min(right_down[1], grey_right_down[1])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (right_down[0] - left_up[0] + 1) * (right_down[1] - left_up[1] + 1)
    boxBArea = (grey_right_down[0] - grey_left_up[0] + 1) * (grey_right_down[1] - grey_left_up[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    print("IOU: ", iou)
    return iou


def imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance, yaw, pit, IOU=None):
    '''
    @brief: 解算信息可视化
    @param: 检测结果图片，检测延迟，帧ID，armor center, 距离， 角度
    @return:
    '''
    armor_cent_x = int(armor_cent[0])
    armor_cent_y = int(armor_cent[1])
    # delt_t = "detect cost: " + str(np.round(detect_delay_time, 3)) + "ms"
    delt_t = "detect cost: " + str(int(detect_delay_time)) + "ms---" + str(frame_id)
    cv2.putText(vis_img, delt_t, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    # armor cent circle
    cv2.circle(vis_img, (armor_cent_x, armor_cent_y), 3, (255, 255, 255), -1)
    # armor cent pre circle
    # cv2.circle(vis_img, (int(pre_x), int(pre_y)), 3, (0, 255, 0), -1)

    angle_t = "AY: " + str(np.around(yaw, 2)) + "--P: " + str(np.around(pit, 2))
    dis_t = "AD: " + str(np.around(distance, 3))
    # print("PRE: IOU -- ", IOU)
    if IOU is not None:
        if IOU > 0.3:
            IOU_t = "Same: " + str(np.around(IOU, 2))
        else:
            IOU_t = "Diff: " + str(np.around(IOU, 2))
    else:
        IOU_t = "NewArmor"
    cv2.putText(vis_img, IOU_t, (armor_cent_x, armor_cent_y + 40),
                cv2.FONT_HERSHEY_PLAIN,
                0.75, (0, 255, 0))

    cv2.putText(vis_img, angle_t, (armor_cent_x, armor_cent_y),
                cv2.FONT_HERSHEY_PLAIN,
                0.75, (0, 255, 0))
    cv2.putText(vis_img, dis_t, (armor_cent_x, armor_cent_y + 20),
                cv2.FONT_HERSHEY_PLAIN,
                0.75, (0, 255, 0))

    # if pre_box is not None:
    #     pre_box[0] = tuple(map(int, pre_box[0]))
    #     pre_box[1] = tuple(map(int, pre_box[1]))
    #     cv2.rectangle(vis_img, pre_box[0], pre_box[1], color=(0, 0, 255), thickness=1)

    return vis_img


def solve_armor_pos(armor_point_list):
    '''
    @brief: 解算armor位姿
    @param: armor二点坐标
    @return: 角度，距离，armor center
    '''
    # 装甲板坐标顺时针排序
    left_up, right_up, right_down, left_down = point_sort(armor_point_list)
    # 计算装甲板中心
    armor_cent = box_cent(left_up, left_down, right_up)
    # armor_cent_list.append(armor_cent)
    # 装甲板角度解算
    rotation_vector, rotation_matrix, translation_vector, camera_position, yaw, pit, distance = \
        gimbal_pit_yaw(left_up, right_up, right_down, left_down, armor_cent)

    return yaw, pit, distance, armor_cent


def send_msg_to_COM(ser, id, delay_time, yaw, pit, distance):
    '''
    @brief: 发送角度信息到串口
    @param: 串口对象，帧ID，延迟，角度
    @return: 写入数据长度
    '''
    if abs(yaw) > 25 or abs(pit) > 25:
        yaw = 0
        pit = 0
    # Pitch安装电机向上为正，向下为负
    pit = pit * 1
    # pit = 0
    yaw = yaw * 1
    msg = pack_COM_data(id, delay_time, yaw, pit, distance)
    rst = ser.write(msg)
    return rst


def select_attack_armor(last_attack_armor, new_armor):
    '''
    @brief: 获取两个装甲板四点坐标的IOU
    @param: 上一次击打的Armor和当前Armor的二点坐标
    @return: IOU
    '''
    # 由左上，右下二点生成左上，右上，右下，左下四点
    last_left_up, _, last_right_down, _ = point_sort(last_attack_armor)
    left_up, right_up, right_down, left_down = point_sort(new_armor)
    # 计算IOU
    IOU = get_IOU(last_left_up, last_right_down, left_up, right_down)
    return IOU


def get_pre_box(bx_list, armor_cent, pre_x, pre_y):
    delt_x = pre_x - armor_cent[0]
    delt_y = pre_y - armor_cent[1]

    for i in bx_list:
        i[0] = i[0] + delt_x
        i[1] = i[1] + delt_y

    return bx_list


def get_pre_angle(yaw, pit, detect_delay_time, dis, last_attack_angle=None, reset_flag=False):
    '''
    @brief: 通过卡尔曼滤波获取预测角度
    @param: yaw, pit角度和检测延迟
    @return: 预测得到的角度和速度
    '''
    if last_attack_angle is not None:
        # 使用pnp解算的速度
        yaw_speed = (yaw - last_attack_angle[0])
        pit_speed = (pit - last_attack_angle[1])
        # 使用卡尔曼预测的速度
        # yaw_speed = last_attack_angle[2]
        # pit_speed = last_attack_angle[3]
    else:
        # todo：速度最好依靠陀螺仪数据
        yaw_speed = 0
        pit_speed = 0

    # 目标切换，重置卡尔曼参数
    if reset_flag:
        # yaw_KF.reset()
        yaw_speed = 0
        pit_speed = 0
    pre_yaw, pre_yaw_s = yaw_KF.track(yaw, yaw_speed, detect_delay_time)
    # pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = bx_KF_pre.track(yaw, pit, yaw_speed, pit_speed, detect_delay_time)
    pre_pit, pre_pit_s = pit_KF.track(pit, pit_speed, detect_delay_time)
    # return pre_yaw, pre_pit, pre_yaw_s, pre_pit_s
    # 使用pnp解算的差值返回速度
    yaw_delay = shoot_delay + dis / bullet_v
    yaw_compensate = np.arctan(yaw_speed * yaw_delay / dis)
    pre_yaw += yaw_compensate
    LOGGING.info("Yaw shoot compensate:{}".format(yaw_compensate))
    return pre_yaw, pre_pit, yaw_speed, pit_speed


def pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, yaw_s, pit_s):
    LOGGING.info("PRE: Frame:{}, Armor Yaw:{},Pit:{}, Delt yaw:{}, Delt Pit:{}, Yaw_s:{}, Pit_s:{}".format(frame_id, yaw, pit, pre_yaw - yaw, pre_pit - pit, yaw_s, pit_s))
    


def KFpredictor_main():
    try:
        notarget_frame = 0
        frame_id = 0
        last_frame_id = 0

        vis_img = None
        video_out = None

        # 上一次击打装甲板信息
        last_attack_armor = None
        last_attack_angle = None
        # armor_cent_list = []
        # car_cent_list = []
        # last_armor_cent = [0, 0]
        # last_pre_armor_cent = [0, 0]
        # 攻击次数
        attack_times = 0
        detect_delay_time = 0

        if Config.WRITE_VIDEO:
            filename = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # 视频存储
            video_out = cv2.VideoWriter(filename, fourcc, 60, (416, 416))
        if Config.CVMSG2COM:
            # 修改信息： 发送编号，接收编号
            Config.ser_obj = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

        # ------- JUST DEBUG INGO -------
        if PRE_DEBUG_INFO:
            yaw_angle_file = time_fmt + "yaw.txt"
            # pre_yaw_angle = time_fmt + "pre_armor_cent.txt"
            yaw_f = open(yaw_angle_file, "w+")
            # pre_yaw_f = open(pre_yaw_angle, "w+")

        while True:

            pre_start_t = time.time()

            if not Q_detector2predictor.empty():
                notarget_frame = 0

                detect_data = Q_detector2predictor.get()
                # 存活装甲板列表
                armor_list = detect_data[0]
                # 车辆列表
                car_list = detect_data[1]
                # 灰色装甲板列表
                grey_armor_list = detect_data[2]
                # frame id
                detect_id = detect_data[3]
                # 检测延迟
                detect_delay_time = detect_data[4]
                # 检测结果可视化
                vis_img = detect_data[5]

                if frame_id > detect_id:
                    print("DETECT2PRE queue size:{}".format(Q_detector2predictor.qsize()))
                    continue

                frame_id = detect_id

                # 检测到存活装甲板
                if armor_list:
                    # 只有一块装甲板，直接解算
                    if len(armor_list) == 1:
                        yaw, pit, distance, armor_cent = solve_armor_pos(armor_list[0])
                        pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = get_pre_angle(yaw, pit, detect_delay_time, distance, last_attack_angle)
                        # 控制输出日志数量
                        if frame_id - last_frame_id > log_cnt:
                            pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                            last_frame_id = frame_id
                        if Config.CVMSG2COM:
                            rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                            if rst:
                                LOGGING.info("PRE:{}, write one armor--yaw:{}, pit:{}, len:{} to COM!".format(frame_id, yaw, pit, rst))


                        # 记录历史数据
                        last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                        last_attack_armor = armor_list[0]

                        # ------ VIS INFO -----
                        if vis_img is not None:
                            vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance, yaw, pit, IOU=None)

                    # 多块装甲板，面积筛选和角度筛选
                    elif len(armor_list) > 1:
                        # 面积降序
                        armor_list.sort(key=box_area, reverse=True)
                        if last_attack_armor is not None and last_attack_angle is not None:
                            IOU = select_attack_armor(last_attack_armor, armor_list[0])
                            yaw, pit, distance, armor_cent = solve_armor_pos(armor_list[0])
                            # IOU > 0.3， 同一块装甲板, 解算并预测
                            if IOU > 0.3:
                                pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = get_pre_angle(yaw, pit, detect_delay_time, distance, last_attack_angle)

                                # 控制输出日志数量
                                if frame_id - last_frame_id > log_cnt:
                                    pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                                    last_frame_id = frame_id

                                if Config.CVMSG2COM:
                                    rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                                    if rst:
                                        LOGGING.info("Write same armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))
                                # 记录历史数据
                                last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                                last_attack_armor = armor_list[0]

                                # ------ VIS INFO -----
                                if vis_img is not None:
                                    vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance, yaw,
                                                          pit, IOU=IOU)

                            # IOU<0.4，不同装甲板，筛选是否需要切换目标
                            else:
                                # 面积和角度筛选, 条件成立为目标切换
                                # cv2.imwrite(str(frame_id) + "iou0.jpg",vis_img)
                                # TODO: 1.2系数不严谨，小陀螺模式下切换目标比较慢,小陀螺模式要单独编写
                                # if (box_area(armor_list[0]) / box_area(last_attack_armor)) > 1.1 and \
                                #         abs(yaw - last_attack_angle[0]) < 15:
                                if abs(yaw - last_attack_angle[0]) < 18:
                                    # 目标切换，重置卡尔曼滤波
                                    pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = get_pre_angle(yaw, pit, detect_delay_time, distance, last_attack_angle=None, reset_flag=True)
                                    print("IOU -- , new target", IOU)

                                    # 控制输出日志数量
                                    if frame_id - last_frame_id > log_cnt:
                                        pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                                        last_frame_id = frame_id

                                    if Config.CVMSG2COM:
                                        rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                                        if rst:
                                            LOGGING.info(
                                                "Write diff armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))

                                    # 记录历史数据
                                    last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                                    last_attack_armor = armor_list[0]

                                    # ------ VIS INFO -----
                                    if vis_img is not None:
                                        vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance,
                                                              yaw, pit, IOU=None)
                                # 条件不成立，对上一次角度预测
                                else:
                                    # 对当前的角度作预测
                                    pre_yaw, pre_yaw_s = yaw_KF.track(last_attack_angle[0], last_attack_angle[2], detect_delay_time)
                                    pre_pit, pre_pit_s = pit_KF.track(last_attack_angle[1], last_attack_angle[3], detect_delay_time)
                                    # pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = bx_KF_pre.track(last_attack_angle[0],
                                    #                                                          last_attack_angle[1],
                                    #                                                          last_attack_angle[2],
                                    #                                                          last_attack_angle[3],
                                    #                                                          detect_delay_time)
                                    # 控制输出日志数量
                                    if frame_id - last_frame_id > log_cnt:
                                        pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                                        last_frame_id = frame_id

                                    if Config.CVMSG2COM:
                                        rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                                        if rst:
                                            LOGGING.info(
                                                "Write last armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))

                                    # 记录历史数据
                                    last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                                    last_attack_armor = armor_list[0]

                                    # ------ VIS INFO -----
                                    if vis_img is not None:
                                        vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance,
                                                              yaw, pit, IOU=IOU)

                        # 第一帧同时出现多块装甲板，选取面积最大的击打
                        else:
                            yaw, pit, distance, armor_cent = solve_armor_pos(armor_list[0])
                            pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = get_pre_angle(yaw, pit, detect_delay_time, distance, last_attack_angle)

                            # 控制输出日志数量
                            if frame_id - last_frame_id > log_cnt:
                                pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                                last_frame_id = frame_id

                            if Config.CVMSG2COM:
                                rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                                if rst:
                                    LOGGING.info("Write multi armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))
                            # 记录历史数据
                            last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                            last_attack_armor = armor_list[0]

                            # ------ VIS INFO -----
                            if vis_img is not None:
                                vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance, yaw,
                                                      pit, IOU=None)

                    # -----DEBUG INGO -----
                    if PRE_DEBUG_INFO:
                        yaw_f.write(str(frame_id) + " " + str(yaw) + " " + str(pre_yaw) + " " + str(pre_yaw_s) + "\n")


                # TODO: solvePoint2Angle()考虑过重构，但是不同的point处理逻辑不同
                # 如果有灰色装甲板而且存在上一次击打的装甲板，计算IOU判断是否同一块
                elif (last_attack_armor is not None and last_attack_angle is not None) and grey_armor_list:
                    # 灰色装甲板面积排序
                    grey_armor_list.sort(key=box_area, reverse=True)
                    yaw, pit, distance, armor_cent = solve_armor_pos(grey_armor_list[0])
                    IOU = select_attack_armor(last_attack_armor, grey_armor_list[0])
                    # IOU > 阈值，击打同一块装甲板
                    if IOU > 0.3:
                        if attack_times <= 12:
                            LOGGING.info("Attack times: {}".format(attack_times))
                            # 攻击次数小于35，持续击打
                            pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = get_pre_angle(yaw, pit, detect_delay_time, distance, last_attack_angle)

                            # 控制输出日志数量
                            if frame_id - last_frame_id > log_cnt:
                                pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)

                                last_frame_id = frame_id

                            if Config.CVMSG2COM:
                                rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                                if rst:
                                    LOGGING.info("Write grey armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))

                            # 记录历史数据
                            last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                            last_attack_armor = grey_armor_list[0]
                            # 攻击次数加1
                            attack_times += 1

                            # ------ VIS INFO -----
                            if vis_img is not None:
                                vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance, yaw,
                                                      pit, IOU=IOU)
                        else:
                            attack_times = 0
                            last_attack_armor = None
                            last_attack_angle = None
                    else:
                        # 击打上次装甲板
                        pre_yaw, pre_yaw_s = yaw_KF.track(last_attack_angle[0], last_attack_angle[2], detect_delay_time)
                        pre_pit, pre_pit_s = pit_KF.track(last_attack_angle[1], last_attack_angle[3], detect_delay_time)
                        # pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = bx_KF_pre.track(last_attack_angle[0],
                        #                                                          last_attack_angle[1],
                        #                                                          last_attack_angle[2],
                        #                                                          last_attack_angle[3],
                        #                                                          detect_delay_time)

                        # 控制输出日志数量
                        if frame_id - last_frame_id > log_cnt:
                            pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)

                            last_frame_id = frame_id

                        if Config.CVMSG2COM:
                            rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw, pre_pit, distance)
                            if rst:
                                LOGGING.info("Write last armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))
                        # 记录历史数据, 仅更新角度值
                        last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                        # last_attack_armor = last_attack_armor

                        # ------ VIS INFO -----
                        if vis_img is not None:
                            vis_img = imginfo_vis(vis_img, detect_delay_time, frame_id, armor_cent, distance,
                                                  yaw, pit, IOU=None)

                pre_cost = int((time.time() - pre_start_t) * 1000)
                LOGGING.info("PRE: frame: {}, cost: {}".format(frame_id, str(pre_cost)))

            else:
                notarget_frame += 1
                # if (last_attack_armor is not None) and (last_attack_angle is not None) and (notarget_frame < 3) :
                if False:
                    # 对当前的角度作预测
                    pre_yaw, pre_yaw_s = yaw_KF.track(last_attack_angle[0], last_attack_angle[2], detect_delay_time)
                    pre_pit, pre_pit_s = pit_KF.track(last_attack_angle[1], last_attack_angle[3], detect_delay_time)

                    # pre_yaw, pre_pit, pre_yaw_s, pre_pit_s = bx_KF_pre.track(last_attack_angle[0],
                    #                                                          last_attack_angle[1],
                    #                                                          last_attack_angle[2],
                    #                                                          last_attack_angle[3],
                    #                                                          detect_delay_time)
                    # 控制输出日志数量
                    #if frame_id - last_frame_id > log_cnt:
                    #    pre_log(frame_id, yaw, pit, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s)
                    #    last_frame_id = frame_id
                    if abs(pre_yaw) < 15 or abs(pre_pit) < 15:
                        LOGGING.info("PRE: Wait DETOR Armor Frame:{}, Yaw:{},Pit:{},Yaw_s:{}, Pit_s:{}".format(frame_id, pre_yaw, pre_pit, pre_yaw_s, pre_pit_s))

                        if Config.CVMSG2COM:
                            rst = send_msg_to_COM(Config.ser_obj, frame_id, detect_delay_time, pre_yaw*0.02, 0, 0)
                            if rst:
                                LOGGING.info(
                                    "Write last armor--yaw:{}, pit:{}, len:{} to COM!".format(yaw, pit, rst))

                        # 记录历史数据
                        #last_attack_angle = [yaw, pit, pre_yaw_s, pre_pit_s]
                        #last_attack_armor = armor_list[0]

                    time.sleep(0.010)

                if notarget_frame == 1000:
                    LOGGING.info("No target frame > 1000, Waiting detector result, send 0 angle to COM!")
                    # print("predict frame id:", frame_id, "PRE cost: ", (time.time() - pre_start_t) * 1000)
                    notarget_frame = 0
                    time.sleep(0.100)

            if vis_img is not None and Config.IMG_INFO:
                    cv2.imshow("PREimage", vis_img)
                    k = cv2.waitKey(1)
                    if k == ord("q"):
                        exit(0)
            if vis_img is not None and Config.WRITE_VIDEO:
                video_out.write(vis_img)

            if Config.TEST_VIDEO_FRAME:
                if frame_id == Config.TEST_VIDEO_FRAME:
                    LOGGING.info("Predictor task done, return!")
                    return

    except:
        if Config.CONSOLE_INFO:
            traceback.print_exc()
        if Config.LOG_FILE_INFO:
            traceback.print_exc(file=open(log_file_path, "a+"))

