import cv2

import sys
import time
import traceback
import numpy as np
from common import Config
from common.logFile import LOGGING, log_file_path
from common.msgQueue import Q_camera2detector

def video_test_main():
    # 不同的视频场景测试
    # video_path = "../ModuleTest/cameraTest/blue_red_car.avi"
    # video_path = "./ModuleTest/cameraTest/dis_angle.avi"
    # video_path = "./ModuleTest/cameraTest/KF_exp.avi"
    # video_path = "./ModuleTest/cameraTest/infantry_test01.avi"
    # video_path = "./ModuleTest/cameraTest/watcher.avi"
    # video_path = "./ModuleTest/cameraTest/red_grey_car.avi"
    # video_path = "./ModuleTest/cameraTest/spin.mp4"
    # video_path = "./ModuleTest/cameraTest/blue_spin.avi"
    video_path = "./ModuleTest/cameraTest/two_armors.avi"
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