# -*- coding:utf-8 -*-
# Data: 2021/11/19 下午11:21
import cv2

import sys
import time
import traceback
import numpy as np
from camera import cameraSDK as gx
#import gxipy as gx
from common import Config
from common.logFile import LOGGING, log_file_path
from common.msgQueue import Q_camera2detector


def init_camera():
    # ----- init camera -----
    # 打开设备
    # 枚举设备
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print("ERROR: Camera devices is NONE!")
        sys.exit(-1)
    # 获取设备基本信息列表
    str_sn = dev_info_list[0].get("sn")
    print("Device num:{num}".format(num=dev_num), str_sn)
    # 通过序列号打开设备
    cam = device_manager.open_device_by_sn(str_sn)
    # 帧率
    fps = cam.AcquisitionFrameRate.get()
    # 视频的宽高
    size = (cam.Width.get(), cam.Height.get())
    print("img param: ", cam.Width.get(), cam.Height.get(), fps)
    return cam


write_video = False
def collect_img_main():
    try:
        img_id = 0
        # ----- init camera -----
        # 打开设备, 枚举设备
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            LOGGING.error("CAM ERROR: Camera devices is NONE!")
            # sys.exit(-1)
        # 获取设备基本信息列表
        str_sn = dev_info_list[0].get("sn")
        LOGGING.info("CAM: Device num:{}, device:{}".format(dev_num, str_sn))
        # 通过序列号打开设备
        cam = device_manager.open_device_by_sn(str_sn)
        cam.BalanceWhiteAuto.set(2)  # once 单次
        cam.AcquisitionMode.set(2) # continue
        cam.ExposureTime.set(10000) # us

        if not cam:
            LOGGING.error("CAM ERROR, Can not init camera obj!")
            # return -1
        # 帧率
        # fps = cam.CurrentAcquisitionFrameRate.get()
        # 视频的宽高
        size = (cam.Width.get(), cam.Height.get())
        LOGGING.info("CAM: img size{}".format(size))
        cam.stream_on()

        # 开始采集
        while True:
            camera_start_t = time.time()
            raw_image = cam.data_stream[0].get_image()  # 使用相机采集一张图片
            if raw_image is None:
                LOGGING.error("CAM ERROR: Camera raw img is NONE!")
                # raise Exception("RAW IMAGE ERROR")
                continue
            rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取 RGB 图像
            if rgb_image is None:
                LOGGING.error("CAM ERROR: convert raw to rgb image error!")
                # raise Exception("RAW2RGB IMAGE ERROR")
                continue
            numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
            resize_cost = (time.time() - camera_start_t) * 1000
            print("numpy cost", resize_cost)
            numpy_image = cv2.resize(numpy_image, (416, 416))
            resize_cost = (time.time() - camera_start_t) * 1000
            print("resize cost", resize_cost)
            # ----- DEBUG test ------
            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像， 讲RGB转为BGR
            # cv2.imshow("camera img", numpy_image)
            # cv2.waitKey(1)
            # cv2.imwrite('cameraimg.png', numpy_image)

            img_id += 1

            # 原来计划在检测模块判断full()然后clear()，但是如果检测模块崩了就会导致队列被无限填充，所以相机线程也需要判断
            if Q_camera2detector.full():
                Q_camera2detector.queue.clear()
                LOGGING.warning("CAM: Q_camera2detect Queue full, wait task done!")
                # time.sleep(1)
            else:
                Q_camera2detector.put([numpy_image, img_id, camera_start_t])
                if img_id % 2000 == 0:
                    LOGGING.info("CAM2PRE id:{}, Q size:{}".format(img_id, Q_camera2detector.qsize()))

            camera_delt_t = (time.time() - camera_start_t)*1000
            # 测试：每5ms采集一次
            print("CAM: img id: ", img_id, "camera cost:", camera_delt_t)
            #time.sleep(0.003)
            if Config.TEST_VIDEO_FRAME:
                if img_id == Config.TEST_VIDEO_FRAME:
                    return

        # if Config.WRITE_VIDEO:
        #     # ----- 采集视频 ------
        #     # 视频存储的格式
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     # 文件名定义
        #     filename = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.avi'
        #     # 视频存储
        #     out = cv2.VideoWriter(filename, fourcc, 60, size)
        #     for i in range(1200):
        #         raw_image = cam.data_stream[0].get_image()  # 使用相机采集一张图片
        #         rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取 RGB 图像
        #         if rgb_image is None:
        #             continue
        #         numpy_image = rgb_image.get_numpy_array()  # 从 RGB 图像数据创建 numpy 数组
        #         numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像， 讲RGB转为BGR
        #         cv2.namedWindow('video', cv2.WINDOW_NORMAL)  # 创建一个名为video的窗口
        #         cv2.imshow('video', numpy_image)  # 将捕捉到的图像在video窗口显示
        #         out.write(numpy_image)  # 将捕捉到的图像存储

    except:
        if Config.CONSOLE_INFO:
            traceback.print_exc()
        if Config.LOG_FILE_INFO:
            traceback.print_exc(file=open(log_file_path, "a+"))


def video_test_main():
    # 不同的视频场景测试
    # video_path = "../ModuleTest/cameraTest/blue_red_car.avi"
    # video_path = "./ModuleTest/cameraTest/dis_angle.avi"
    video_path = "../ModuleTest/cameraTest/KF_exp.avi"
    # video_path = "./ModuleTest/cameraTest/infantry_test01.avi"
    # video_path = "./ModuleTest/cameraTest/watcher.avi"
    # video_path = "./ModuleTest/cameraTest/red_grey_car.avi"
    # video_path = "./ModuleTest/cameraTest/spin.mp4"
    # video_path = "./ModuleTest/cameraTest/blue_spin.avi"
    # video_path = "./ModuleTest/cameraTest/two_armors.avi"
    # video_path = "./ModuleTest/cameraTest/attack_grey.avi"
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        exit(-1)
    img_id = 0

    while (cap.isOpened()):
        # just for DBUG
        if Config.TEST_VIDEO_FRAME and img_id >= Config.TEST_VIDEO_FRAME:
            LOGGING.info("Camera task done, wait all task done!")
            time.sleep(1)

        ret, rgb_image = cap.read()
        cam_start_t = time.time()
        rgb_image = cv2.resize(rgb_image, (416, 416)).astype(np.uint8)
        img_id += 1
        if rgb_image is None:
            print("ERROR: Video image is NONE, return!")
            return
            # continue
        # ----- DEBUG test ------
        # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像， 讲RGB转为BGR
        # cv2.imshow("VideoInputImg", rgb_image)
        # cv2.imshow("VideoInputImg", bgr_image)
        # cv2.waitKey(1)
        # cv2.imwrite('cameraimg.png', rgb_image)

        # 原来计划在检测模块判断full()然后clear()，但是如果检测模块崩了就会导致队列被无限填充，所以相机线程也需要判断
        if Q_camera2detector.full():
            LOGGING.warning("CAM: Q_camera2detect Queue full, wait task done!")
            Q_camera2detector.queue.clear()
        else:
            Q_camera2detector.put([rgb_image, img_id, cam_start_t])

        camera_delt_t = int((time.time() - cam_start_t) * 1000)
        # 测试：每10ms采集一次
        print("CAM: img id: ", img_id, "camera cost:", camera_delt_t)
        # 测试：每10ms采集一次
        if img_id % 1000 == 0:
            print("video img id:", img_id)
        time.sleep(0.010)
        if Config.TEST_VIDEO_FRAME:
            if img_id == Config.TEST_VIDEO_FRAME:
                LOGGING.info("DETOR: Detector task done, wait all task done!")
                # time.sleep(1)
                return

            # DEBUG test
if __name__ == "__main__":
    # collect_img_main()
    video_test_main()
