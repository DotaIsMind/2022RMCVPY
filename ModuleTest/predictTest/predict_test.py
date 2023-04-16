# -*- coding:utf-8 -*-
# Data: 2021/11/27 下午7:38


from detector.detectorModule import detector_main, video_test_main
from common.msgQueue import Q_camera2detector, Q_detector2predictor

import cv2

video_path = "../cameraTest/angle_t.png"
rgb_image = cv2.imread(video_path)

Q_camera2detector.put(rgb_image)

if __name__ == "__main__":
    detector_main()
