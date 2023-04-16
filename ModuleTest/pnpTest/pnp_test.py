# -*- coding:utf-8 -*-
# Data: 2021/11/30 下午8:27

import numpy as np
import cv2
import time
import sys
from camera.cameraSDK.gxiapi import *


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
    # camera_matrix = np.array([[1290.5269040172795485 * 0.6, 0.0000000000000000, 208.0000000000000000],
    #                           [0.0000000000000000, 1286.5137262745745375 * 0.6, 208.0000000000000000],
    #                           [0, 0, 1]], dtype="double")
    camera_matrix = np.array([[766.5269040172795485, 0.0000000000000000, 208.0000000000000000],
                              [0.0000000000000000, 766.5137262745745375, 208.0000000000000000],
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
    # 坐标系y轴朝下，Pitch朝上旋转为正
    pit_c = -(np.arctan2(translation_vector[1], translation_vector[2]) * 180 / np.pi) + 5.3
    distance = translation_vector[2] / 1000 # 单位:m
    # 这里得到的是相机在装甲板世界坐标系的位姿
    camera_position = -np.matrix(rotation_matrix).T * np.matrix(translation_vector)

    return rotation_vector, rotation_matrix, translation_vector, camera_position, yaw_c[0], pit_c[0], distance


ix, iy = -1, -1  # 初始化鼠标位置

def onmouse(event, x, y, flags, param):  # 创建回调函数
    global ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键
        ix, iy = x, y  # 赋予按下时的鼠标坐标
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(numpy_image, (x, y), 3, (0, 0, 255), -1)  # 当模式为False时画线

# ----- init camera -----
# 打开设备, 枚举设备
device_manager = DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    print("ERROR: Camera devices is NONE!")
    sys.exit(-1)
# 获取设备基本信息列表
str_sn = dev_info_list[0].get("sn")
print("Device num:{num}".format(num=dev_num), str_sn)
# 通过序列号打开设备
cam = device_manager.open_device_by_sn(str_sn)
cam.BalanceWhiteAuto.set(2)  # once 单次

if not cam:
    print("ERROR, Cant init camera obj!")
    exit(-1)
# 帧率
# fps = cam.CurrentAcquisitionFrameRate.get()
# 视频的宽高
size = (cam.Width.get(), cam.Height.get())
print("img size{}, fps: {}", size)
# cam.BalanceWhiteAuto()
cam.stream_on()
# 开始采集
# while True:
raw_image = cam.data_stream[0].get_image()  # 使用相机采集一张图片
rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取 RGB 图像
if rgb_image is None:
    print("ERROR: Camera collect img is NONE!")
    exit(-1)
numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
resize_img = cv2.resize(numpy_image, (416, 416))
# ----- DEBUG test ------
numpy_image = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像， 讲RGB转为BGR
cv2.imwrite("test.png", numpy_image)

# numpy_image = cv2.imread("test.png")
cv2.namedWindow('image')
cv2.setMouseCallback('image', onmouse)

pos_list = []
for i in range(5):
    cv2.imshow("image", numpy_image)
    print("Pos: ", ix, iy)
    k = cv2.waitKey()
    if k == ord('q'):  # 直到按键盘上的'q'键才退出图像
        pos_list.append([ix, iy])
        continue
cv2.destroyAllWindows()

# rotation_vector, translation_vector,distance, yaw, pitch  = gimbal_pit_yaw(pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4])# distance单位换算到米 gimbal_pit_yaw()
# normal armor
# high = 120  # mm
# width = 130  # mm

# big armor
high = 127  # mm
width = 230  # mm
# 世界坐标系：以装甲板中心为原点,x轴向右，y轴向下的坐标系, 分别为左上，右上，右下，左下，单位：mm
object_3d_points = np.array([
    (-width / 2, -high / 2, 0),
    (width / 2, -high / 2, 0),
    (width / 2, high / 2, 0),
    (-width / 2, high / 2, 0),
    (0, 0, 0)
    # (0, 0, 0)
], dtype=np.double)
object_2d_point = np.array((pos_list[0], pos_list[1], pos_list[2], pos_list[3], pos_list[4]), dtype=np.double)
img_x_resize_factor = np.true_divide(416, 1280)
img_y_resize_factor = np.true_divide(416, 1024)
camera_matrix = np.array(
    [[1290.5269040172795485 * img_x_resize_factor, 0.0000000000000000, 643.0843358495335451  * img_x_resize_factor],
     [0.0000000000000000, 1286.5137262745745375 * img_y_resize_factor, 492.8023899252162892 * img_y_resize_factor],
     [0.0000000000000000, 0.0000000000000000, 1.0000000000000000]], dtype="double")


# 畸变参数
dist_coefs = np.transpose(
    [-0.2235773472428870, 0.2361982138837830, 0.0000000000000000, 0.0000000000000000, -0.0145952258080805])
# 求解相机位姿
found, rotation_vector, translation_vector = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)

# solvePnP得到是旋转向量，使用罗格里德转化为旋转矩阵
rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
print("rotation vector norm: ", np.linalg.norm(rotation_vector))
# 偏移矩阵就是目标在相机坐标系下的坐标
# yaw_c: atan(x/z)  pit_c: atan(y/sqrt(x^2+z^2))
yaw_c = np.arctan2(translation_vector[0], translation_vector[2]) * 180 / np.pi
# 坐标系y轴朝下，Pitch朝上旋转为正
pit_c = np.arctan2(translation_vector[1], np.sqrt(translation_vector[0] ** 2 + translation_vector[2] ** 2)) * 180 / np.pi
distance = np.linalg.norm(translation_vector)  / 1000# 单位:m
print("Translation yaw:{}, pit:{}, dis: {}".format(yaw_c, pit_c, distance))


# Pc = R*Pw + T，令Pc=0，0 = R*Pw + T, Pw = -T * R^t (正交矩阵的逆等于转置）可以得到相机坐标系下的坐标
camera_postion = - np.matrix(rotation_matrix).T * np.matrix(translation_vector)
# 相机所在位置相对于世界坐标系（装甲板坐标系）的坐标
# todo:所得是相机位姿，需要的是目标在相机坐标系下的位姿, 偏移矩阵就是？
print("camera pos in world cordination: ", camera_postion)
# # yaw = actan(x/z)
# yaw = np.arctan2(camera_postion[0], camera_postion[2]) * 180/np.pi
# # pit = actan(y/sqrt(x**2+z**2))
# pit1 = np.arctan2(camera_postion[1], np.sqrt(camera_postion[0] ** 2, camera_postion[2] ** 2)) * 180/np.pi
# # pit = actan(y/z)
# pit2 = np.arctan2(camera_postion[1], camera_postion[2]) * 180/np.pi
# print("Camera pos in world cor Y-P: ", yaw, pit1, pit2, "pit error: ", pit2-pit1)


# 验证根据博客http://www.cnblogs.com/singlex/p/pose_estimation_1.html提供方法求解相机位姿
# # 计算相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。旋转顺序z,y,x
# thetaZ = np.arctan2(rotM[1, 0], rotM[0, 0])*180.0/np.pi
# thetaY = np.arctan2(-1.0*rotM[2, 0], np.sqrt(rotM[2, 1]**2 + rotM[2, 2]**2))*180.0/np.pi
# thetaX = np.arctan2(rotM[2, 1], rotM[2, 2])*180.0/np.pi
# print(thetaX, thetaY, thetaZ)
# # 相机坐标系下值
# x = tvec[0]
# y = tvec[1]
# z = tvec[2]
# print("x-y-z", x, y, z)
# # 进行三次旋转
# def RotateByZ(Cx, Cy, thetaZ):
#     rz = thetaZ*np.pi/180.0
#     outX = np.cos(rz)*Cx - np.sin(rz)*Cy
#     outY = np.sin(rz)*Cx + np.cos(rz)*Cy
#     return outX, outY
# def RotateByY(Cx, Cz, thetaY):
#     ry = thetaY*np.pi/180.0
#     outZ = np.cos(ry)*Cz - np.sin(ry)*Cx
#     outX = np.sin(ry)*Cz + np.cos(ry)*Cx
#     return outX, outZ
# def RotateByX(Cy, Cz, thetaX):
#     rx = thetaX*np.pi/180.0
#     outY = np.cos(rx)*Cy - np.sin(rx)*Cz
#     outZ = np.sin(rx)*Cy + np.cos(rx)*Cz
#     return outY, outZ
# (x, y) = RotateByZ(x, y, -1.0*thetaZ)
# (x, z) = RotateByY(x, z, -1.0*thetaY)
# (y, z) = RotateByX(y, z, -1.0*thetaX)
# Cx = x*-1
# Cy = y*-1
# Cz = z*-1
# # 输出相机位置
# print(Cx, Cy, Cz)
# # 输出相机旋转角
# print(thetaX, thetaY, thetaZ)
# 对第五个点进行验证
# Out_matrix = np.concatenate((rotM, tvec), axis=1)
# pixel = np.dot(camera_matrix, Out_matrix)
# pixel1 = np.dot(pixel, np.array([0, 100, 105, 1], dtype=np.double))
# pixel2 = pixel1/pixel1[2]
# print(pixel2)