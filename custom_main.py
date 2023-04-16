# -*- coding:utf-8 -*-
# Data: 2021/11/28 下午2:36

from camera.cameraModule import collect_img_main, video_test_main
from detector.detectorModule import YOLODetector
from predictor.kalmanfilter.TrackKF_2D import KalmanFilter
from common.logFile import LOGGING
from common import Config
import sys
from camera import cameraSDK as gx
import time
import numpy as np
import cv2
import serial
from ctypes import *
import struct


def point_sort(box):
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
    high = 120 #mm
    width = 130 #mm
    # 世界坐标系：以装甲板中心为原点,x轴向右，y轴向下的坐标系, 分别为左上，右上，右下，左下，单位：mm
    model_points = np.array([
        (-width/2, -high/2, 0),
        (width/2, -high/2, 0),
        (width/2, high/2, 0),
        (-width/2, high/2, 0),
        (0, 0, 0)    # 装甲板中心为世界坐标系原点
    ], dtype="double")
    # 大恒相机畸变系数  焦距： 1290.526904017280   1286.513726274575  像素单位  像元尺寸：4.8*4.8微米  实际焦距： 0.2688597716702
    focal_length = np.array([1290.526904017280, 1286.513726274575])
    # camera_matrix = np.array([[1290.5269040172795485, 0.0000000000000000, 643.0843358495335451],
    #                           [0.0000000000000000, 1286.5137262745745375, 492.8023899252162892],
    #                           [0, 0, 1]], dtype="double")
    # camera_matrix = np.array([[1290.5269040172795485, 0.0000000000000000, 640.0000000000000000],
    #                           [0.0000000000000000, 1286.5137262745745375, 512.0000000000000000],
    #                           [0, 0, 1]], dtype="double")
    # resize(416, 416)以后相机像素坐标中心坐标改变
    # todo: 图片resize()以后这个fx, fy也需要调优
    # camera_matrix = np.array([[1290.5269040172795485, 0.0000000000000000, 208.0000000000000000],
    #                           [0.0000000000000000, 1286.5137262745745375, 208.0000000000000000],
    #                           [0, 0, 1]], dtype="double")
    # daheng 小镜头
    # camera_matrix = np.array([[640.5269040172795485, 0.0000000000000000, 208.0000000000000000],
    #                           [0.0000000000000000, 646.5137262745745375, 208.0000000000000000],
    #                           [0, 0, 1]], dtype="double")

    # 大恒大镜头
    camera_matrix = np.array([[460.5269040172795485, 0.0000000000000000, 208.0000000000000000],
                              [0.0000000000000000, 466.5137262745745375, 208.0000000000000000],
                              [0, 0, 1]], dtype="double")
    # 畸变参数
    dist_coeffs = np.transpose(
        [-0.2235773472428870, 0.2361982138837830, 0.0000000000000000, 0.0000000000000000, -0.0145952258080805])
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,  #
                                                                  image_points, camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    # solvePnP得到是旋转向量，使用罗格里德转化为旋转矩阵
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    # 偏移矩阵就是目标在相机坐标系下的坐标
    # yaw_c: atan(x/z)  pit_c: atan(y/z)
    yaw_c = np.arctan2(translation_vector[0], translation_vector[2]) * 180 / np.pi
    pit_c = -(np.arctan2(translation_vector[1], translation_vector[2]) * 180 / np.pi)
    # pit_c = np.arctan2(translation_vector[1], translation_vector[2]) * 180 / np.pi
    distance = translation_vector[2] / 1000 # 单位:m
    # 这里得到的是相机在装甲板世界坐标系的位姿
    camera_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)

    return rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c, pit_c, distance


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
    # delay_time_t = c_float(delay_time).value
    delay_time_t = c_float(0.030).value
    yaw_angle_t = c_float(yaw_angle).value
    pit_angle_t = c_float(pit_angle).value
    # pit_angle_t = c_float(0).value
    cv2COM_info = [flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, flag_t_t]
    # uint8-usigned_char: B, uint32-usigned_int: I, float-float: f
    pack_fmt = "@BIfffB"
    COM_strm = struct.pack(pack_fmt, flag_id_t, id_t, delay_time_t, yaw_angle_t, pit_angle_t, flag_t_t)
    # COM_obj = cv2COM(*cv2COM_info)
    return COM_strm


# if __name__ == "__main__":
#     img_id = 0
#     if Config.WRITE_VIDEO:
#         filename = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         # 视频存储
#         video_out = cv2.VideoWriter(filename, fourcc, 25, (416, 416))
#     # ----- init camera -----
#     # 打开设备, 枚举设备
#     device_manager = gx.DeviceManager()
#     dev_num, dev_info_list = device_manager.update_device_list()
#     if dev_num == 0:
#         print("ERROR: Camera devices is NONE!")
#         sys.exit(-1)
#     # 获取设备基本信息列表
#     str_sn = dev_info_list[0].get("sn")
#     print("Device num:{num}".format(num=dev_num), str_sn)
#     # 通过序列号打开设备
#     cam = device_manager.open_device_by_sn(str_sn)
#     if not cam:
#         print("ERROR, Cant init camera obj!")
#         exit(-1)
#     # 帧率
#     fps = cam.CurrentAcquisitionFrameRate.get()
#     # 视频的宽高
#     size = (cam.Width.get(), cam.Height.get())
#     print("img size{}, fps: {}", size, fps)
#     # cam.BalanceWhiteAuto()
#     cam.stream_on()
#     # 开始采集
#
#     # init detect obj
#     detect_obj = YOLODetector()
#     # init object
#     KF_pre = KalmanFilter()
#     if Config.CVMSG2COM:
#         # 初始化串口对象
#         ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
#
#     frame_id = 0
#
#     armor_cent_list = []
#     car_cent_list = []
#
#     last_armor_angle = [0, 0]
#     last_car_angle = [0, 0]
#     img_id = 0
#
#     while cam:
#         detect_start_t = time.time()
#
#         raw_image = cam.data_stream[0].get_image()  # 使用相机采集一张图片
#         rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取 RGB 图像
#         if rgb_image is None:
#             print("ERROR: Camera collect img is NONE!")
#             continue
#         numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
#         bgr_img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
#         cv2.namedWindow("InputImage", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("InputImage", 640, 480)
#         cv2.moveWindow("InputImage", 0, 0)
#         cv2.imshow("InputImage", bgr_img)
#         cv2.waitKey(1)
#
#         # cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
#         # cv2.imshow("CameraInput", numpy_image)
#         # ----- DEBUG test ------
#         detect_rst, vis_img = detect_obj.detect_result(bgr_img)
#         if detect_rst is None:
#             LOGGING.info("frame id: {frame}, detector rst is NONE, next frame!".format(frame=frame_id))
#             continue
#         # slice final_box, final score, final class index
#         final_boxes, final_scores, final_cls_inds = detect_rst[:, :4], detect_rst[:, 4], detect_rst[:, 5]
#         # 筛选出装甲板列表,车列表,可视化图像
#         armor_list, car_list = detect_obj.select_target(final_boxes, final_scores, final_cls_inds)
#         detect_end_t = time.time()
#         delay_time = detect_end_t - detect_start_t
#         frame_id += 1
#         if Config.IMG_INFO:
#             delt_t = "detect cost: " + str(np.round(delay_time, 3) * 1000) + "ms--frame: " + str(frame_id)
#             cv2.putText(vis_img, delt_t, (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
#
#         # 检测到装甲板
#         if armor_list:
#             # 装甲板面积大小排序, 最大的index=0
#             armor_list.sort(key=box_area, reverse=True)
#
#             # 计算装甲板四个角点坐标,中心和角度解算
#             left_up, right_up, right_down, left_down = point_sort(armor_list[0])
#             armor_cent = box_cent(left_up, left_down, right_up)
#             armor_cent_list.append(armor_cent)
#             rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c, pit_c, distance \
#                 = gimbal_pit_yaw(left_up, right_up, right_down, left_down, armor_cent)
#
#             delt_t = time.time() - detect_start_t
#             print("all cost: ", delt_t)
#             yaw_c, pit_c = KF_pre.track(yaw_c, pit_c)
#
#             if abs(yaw_c) > 20 or abs(pit_c) > 20:
#                 yaw_c = float(0)
#                 pit_c = float(0)
#                 py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c, pit_c)
#                 print("Out vlue--", py2COM_data.hex())
#
#                 LOGGING.info("Angle exception, send 0 angle to COM!")
#             else:
#                 # pack为字节流和串口通信,传到电控模块
#                 py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c - last_armor_angle[0],
#                                             pit_c - last_armor_angle[1])
#                 print("Out vlue--", py2COM_data.hex())
#
#             last_armor_angle = [yaw_c, pit_c]
#             if Config.CVMSG2COM:
#                 # write_com_rst应等于写入数据的长度
#                 write_com_rst = ser.write(py2COM_data)
#             # last_measurement = [pre_x, pre_y]
#
#                 if write_com_rst:
#                     LOGGING.info("Write armor predict data, len {} to COM!".format(len(py2COM_data)))
#                     # LOGGING.info("Armor DELTA x: {}, Y: {}".format(armor_cent_x - last_measurement[0], armor_cent_y-last_measurement[1]))
#                     LOGGING.info("Armor yaw_c: {}, pit_c: {}".format(yaw_c, pit_c))
#                     LOGGING.info("Armor DELTA yaw_c: {}, pit_c: {}".format(np.around(yaw_c - last_armor_angle[0], 3),
#                                                                            np.around(pit_c - last_armor_angle[1], 3)))
#             if vis_img is not None:
#                 cv2.circle(vis_img, (int(armor_cent[0]), int(armor_cent[1])), 3, (255, 255, 255), -1)
#                 angle_t = "Y: " + str(np.around(yaw_c, 2)) + "--P: " + str(np.around(pit_c, 2))
#                 dis_t = "D: " + str(np.around(distance, 3))
#
#                 cv2.putText(vis_img, angle_t, (int(armor_cent[0]), int(armor_cent[1])), cv2.FONT_HERSHEY_PLAIN,
#                             0.75, (0, 255, 0))
#                 cv2.putText(vis_img, dis_t, (int(armor_cent[0]), int(armor_cent[1]) + 30), cv2.FONT_HERSHEY_PLAIN,
#                             0.75, (0, 255, 0))
#
#         # todo： 优先跟踪面积最大的armor_angle_list[0]， 后续根据云台角度选择偏转角度最小的目标
#         elif car_list:
#             # todo : todo: todo: 如果只识别到车， 车的3D点大小和armor的大小不一样，要重新建立3Dpoint
#             car_list.sort(key=box_area, reverse=True)
#             # 计算装甲板四个角点坐标,中心和角度解算
#             left_up, right_up, right_down, left_down = point_sort(car_list[0])
#             car_cent = box_cent(left_up, left_down, right_up)
#             car_cent_list.append(car_cent)
#             rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c, pit_c, distance \
#                 = gimbal_pit_yaw(left_up, right_up, right_down, left_down, car_cent)
#
#             delt_t = time.time() - detect_start_t
#             print("all cost: ", delt_t)
#             yaw_c, pit_c = KF_pre.track(yaw_c, pit_c)
#
#             if abs(yaw_c) > 20 or abs(pit_c) > 20:
#                 yaw_c = float(0)
#                 pit_c = float(0)
#                 py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c, pit_c)
#                 print("Out vlue--", py2COM_data)
#                 LOGGING.info("Angle exception, send 0 angle to COM!")
#             else:
#                 # pack为字节流和串口通信,传到电控模块
#                 py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c - last_car_angle[0],
#                                             pit_c - last_car_angle[1])
#                 print("Out vlue--", py2COM_data)
#
#             last_car_angle = [yaw_c, pit_c]
#             if Config.CVMSG2COM:
#
#                 # write_com_rst应等于写入数据的长度
#                 write_com_rst = ser.write(py2COM_data)
#             # last_measurement = [pre_x, pre_y]
#
#                 if write_com_rst:
#                     LOGGING.info("Write car predict data, len {} to COM!".format(len(py2COM_data)))
#                     LOGGING.info("Car yaw_c: {}, pit_c: {}".format(yaw_c, pit_c))
#                     LOGGING.info("Armor DELTA yaw_c: {}, pit_c: {}".format(np.around(yaw_c - last_car_angle[0], 3),
#                                                                            np.around(pit_c - last_car_angle[1], 3)))
#
#             if vis_img is not None:
#                 cv2.circle(vis_img, (int(car_cent[0]), int(car_cent[1])), 3, (255, 255, 255), -1)
#                 angle_t = "Y: " + str(np.around(yaw_c, 2)) + "--P: " + str(np.around(pit_c, 2))
#                 dis_t = "D: " + str(np.around(distance, 3))
#
#                 cv2.putText(vis_img, angle_t, (int(car_cent[0]), int(car_cent[1])), cv2.FONT_HERSHEY_PLAIN,
#                             0.75, (0, 255, 0))
#                 cv2.putText(vis_img, dis_t, (int(car_cent[0]), int(car_cent[1]) + 30),
#                             cv2.FONT_HERSHEY_PLAIN,
#                             0.75, (0, 255, 0))
#
#         else:
#             # 丢失目标发0
#             # pack为字节流和串口通信,传到电控模块
#             py2COM_data = pack_COM_data(frame_id, delay_time, 0, 0)
#             if Config.CVMSG2COM:
#                 write_com_rst = ser.write(py2COM_data)
#                 if write_com_rst:
#                     LOGGING.info("Have no target, send {} data to COM".format(len(py2COM_data)))
#
#
#         if vis_img is not None and Config.WRITE_VIDEO:
#             video_out.write(vis_img)
#         if vis_img is not None and Config.IMG_INFO:
#             # for armor_cent in armor_cent_list:
#             #     cv2.circle(vis_img, (int(armor_cent[0]), int(armor_cent[1])), 3, (0, 0, 255), -1)
#             #
#             # for car_cent in car_cent_list:
#             #     cv2.circle(vis_img, (int(car_cent[0]), int(car_cent[1])), 3, (0, 0, 255), -1)
#             cv2.imshow("AutoAimMode", vis_img)
#             k = cv2.waitKey(1)
#             if k == ord("q"):
#                 exit(0)
#
#         if Config.WRITE_VIDEO:
#             video_out.write(vis_img)
#
#         # time.sleep(0.03)
#         # if frame_id == Config.TEST_VIDEO_FRAME -1:
#         #     exit(0)


# video test



if __name__ == "__main__":
    video_path = "./ModuleTest/cameraTest/dis_angle.avi"
    # video_path = "./ModuleTest/cameraTest/watcher.avi"
    # video_path = "./ModuleTest/cameraTest/red_grey_car.avi"
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        exit(-1)

    if Config.WRITE_VIDEO:
        filename = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 视频存储
        video_out = cv2.VideoWriter(filename, fourcc, 25, (416, 416))
    # init detect obj
    detect_obj = YOLODetector()
    # init object
    KF_pre = KalmanFilter()
    if Config.CVMSG2COM:
        # 初始化串口对象
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

    frame_id = 0

    armor_cent_list = []
    car_cent_list = []

    last_armor_angle = [0, 0]
    last_car_angle = [0, 0]
    img_id = 0
    while (cap.isOpened()):
        # just for DBUG
        if img_id >= Config.TEST_VIDEO_FRAME:
            LOGGING.info("Video task done, wait all task done!")
            time.sleep(1)
        detect_start_t = time.time()
        ret, rgb_image = cap.read()
        # numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
        # bgr_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("InputImage", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("InputImage", 640, 480)
        cv2.moveWindow("InputImage", 0, 0)
        cv2.imshow("InputImage", rgb_image)
        cv2.waitKey(1)

        # cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("CameraInput", numpy_image)
        # ----- DEBUG test ------
        detect_rst, vis_img = detect_obj.detect_result(rgb_image)
        if detect_rst is None:
            LOGGING.info("frame id: {frame}, detector rst is NONE, next frame!".format(frame=frame_id))
            continue
        # slice final_box, final score, final class index
        final_boxes, final_scores, final_cls_inds = detect_rst[:, :4], detect_rst[:, 4], detect_rst[:, 5]
        # 筛选出装甲板列表,车列表,可视化图像
        armor_list, car_list = detect_obj.select_target(final_boxes, final_scores, final_cls_inds)
        detect_end_t = time.time()
        delay_time = detect_end_t - detect_start_t
        frame_id += 1
        if Config.IMG_INFO:
            delt_t = "detect cost: " + str(np.round(delay_time, 3) * 1000) + "ms--frame: " + str(frame_id)
            cv2.putText(vis_img, delt_t, (25, 25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        # 检测到装甲板
        if armor_list:
            # 装甲板面积大小排序, 最大的index=0
            armor_list.sort(key=box_area, reverse=True)

            # 计算装甲板四个角点坐标,中心和角度解算
            left_up, right_up, right_down, left_down = point_sort(armor_list[0])
            armor_cent = box_cent(left_up, left_down, right_up)
            armor_cent_list.append(armor_cent)
            rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c, pit_c, distance \
                = gimbal_pit_yaw(left_up, right_up, right_down, left_down, armor_cent)

            delt_t = time.time() - detect_start_t
            print("all cost: ", delt_t)
            yaw_c, pit_c = KF_pre.track(yaw_c, pit_c)

            if abs(yaw_c) > 20 or abs(pit_c) > 20:
                yaw_c = float(0)
                pit_c = float(0)
                py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c, pit_c)
                print("Out threshold--", py2COM_data.hex())

                LOGGING.info("Angle exception, send 0 angle to COM!")
            else:
                # pack为字节流和串口通信,传到电控模块
                py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c - last_armor_angle[0],
                                            pit_c - last_armor_angle[1])
                print("Out threshold--", py2COM_data.hex())

            last_armor_angle = [yaw_c, pit_c]
            if Config.CVMSG2COM:
                # write_com_rst应等于写入数据的长度
                write_com_rst = ser.write(py2COM_data)
                # last_measurement = [pre_x, pre_y]

                if write_com_rst:
                    LOGGING.info("Write armor predict data, len {} to COM!".format(len(py2COM_data)))
                    # LOGGING.info("Armor DELTA x: {}, Y: {}".format(armor_cent_x - last_measurement[0], armor_cent_y-last_measurement[1]))
                    LOGGING.info("Armor yaw_c: {}, pit_c: {}".format(yaw_c, pit_c))
                    LOGGING.info("Armor DELTA yaw_c: {}, pit_c: {}".format(np.around(yaw_c - last_armor_angle[0], 3),
                                                                           np.around(pit_c - last_armor_angle[1], 3)))
            if vis_img is not None:
                cv2.circle(vis_img, (int(armor_cent[0]), int(armor_cent[1])), 3, (255, 255, 255), -1)
                angle_t = "Y: " + str(np.around(yaw_c, 2)) + "--P: " + str(np.around(pit_c, 2))
                dis_t = "D: " + str(np.around(distance, 3))

                cv2.putText(vis_img, angle_t, (int(armor_cent[0]), int(armor_cent[1])), cv2.FONT_HERSHEY_PLAIN,
                            0.75, (0, 255, 0))
                cv2.putText(vis_img, dis_t, (int(armor_cent[0]), int(armor_cent[1]) + 30), cv2.FONT_HERSHEY_PLAIN,
                            0.75, (0, 255, 0))

        # todo： 优先跟踪面积最大的armor_angle_list[0]， 后续根据云台角度选择偏转角度最小的目标
        elif car_list:
            # todo : todo: todo: 如果只识别到车， 车的3D点大小和armor的大小不一样，要重新建立3Dpoint
            car_list.sort(key=box_area, reverse=True)
            # 计算装甲板四个角点坐标,中心和角度解算
            left_up, right_up, right_down, left_down = point_sort(car_list[0])
            car_cent = box_cent(left_up, left_down, right_up)
            car_cent_list.append(car_cent)
            rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c, pit_c, distance \
                = gimbal_pit_yaw(left_up, right_up, right_down, left_down, car_cent)

            delt_t = time.time() - detect_start_t
            print("all cost: ", delt_t)
            yaw_c, pit_c = KF_pre.track(yaw_c, pit_c)

            if abs(yaw_c) > 20 or abs(pit_c) > 20:
                yaw_c = float(0)
                pit_c = float(0)
                py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c, pit_c)
                print("Out threshold--", py2COM_data)
                LOGGING.info("Angle exception, send 0 angle to COM!")
            else:
                # pack为字节流和串口通信,传到电控模块
                py2COM_data = pack_COM_data(frame_id, delay_time, yaw_c - last_car_angle[0],
                                            pit_c - last_car_angle[1])
                print("Out threshold--", py2COM_data)

            last_car_angle = [yaw_c, pit_c]
            if Config.CVMSG2COM:

                # write_com_rst应等于写入数据的长度
                write_com_rst = ser.write(py2COM_data)
                # last_measurement = [pre_x, pre_y]

                if write_com_rst:
                    LOGGING.info("Write car predict data, len {} to COM!".format(len(py2COM_data)))
                    LOGGING.info("Car yaw_c: {}, pit_c: {}".format(yaw_c, pit_c))
                    LOGGING.info("Armor DELTA yaw_c: {}, pit_c: {}".format(np.around(yaw_c - last_car_angle[0], 3),
                                                                           np.around(pit_c - last_car_angle[1], 3)))

            if vis_img is not None:
                cv2.circle(vis_img, (int(car_cent[0]), int(car_cent[1])), 3, (255, 255, 255), -1)
                angle_t = "Y: " + str(np.around(yaw_c, 2)) + "--P: " + str(np.around(pit_c, 2))
                dis_t = "D: " + str(np.around(distance, 3))

                cv2.putText(vis_img, angle_t, (int(car_cent[0]), int(car_cent[1])), cv2.FONT_HERSHEY_PLAIN,
                            0.75, (0, 255, 0))
                cv2.putText(vis_img, dis_t, (int(car_cent[0]), int(car_cent[1]) + 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            0.75, (0, 255, 0))

        else:
            # 丢失目标发0
            # pack为字节流和串口通信,传到电控模块
            py2COM_data = pack_COM_data(frame_id, delay_time, 0, 0)
            if Config.CVMSG2COM:
                # write_com_rst应等于写入数据的长度
                write_com_rst = ser.write(py2COM_data)
                if write_com_rst:
                    LOGGING.info("Have no target, send {} data to COM".format(len(py2COM_data)))

        if vis_img is not None and Config.WRITE_VIDEO:
            video_out.write(vis_img)
        if vis_img is not None and Config.IMG_INFO:
            # for armor_cent in armor_cent_list:
            #     cv2.circle(vis_img, (int(armor_cent[0]), int(armor_cent[1])), 3, (0, 0, 255), -1)
            #
            # for car_cent in car_cent_list:
            #     cv2.circle(vis_img, (int(car_cent[0]), int(car_cent[1])), 3, (0, 0, 255), -1)
            cv2.imshow("AutoAimMode", vis_img)
            k = cv2.waitKey(1)
            if k == ord("q"):
                exit(0)

        if Config.WRITE_VIDEO:
            video_out.write(vis_img)

        # time.sleep(0.03)
        # if frame_id == Config.TEST_VIDEO_FRAME -1:
        #     exit(0)


