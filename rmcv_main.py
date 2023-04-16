# -*- coding:utf-8 -*-
# Data: 2021/11/20 上午12:29
import struct

# from camera.cameraModule import collect_img_main
from camera.video_read import video_test_main
from detector.detectorModuleTRT import TRTdetector_main
from predictor.predictorModule import KFpredictor_main
from common.logFile import log_file_path, LOGGING
from common import Config

from OPENCVdetector.config import *
# from OPENCVdetector.armor_detector import ArmorDetector


import time
import serial
import serial.tools.list_ports

import traceback
from threading import Thread

if __name__ == "__main__":
    try:
        if Config.CVMSG2COM:
        # if False:
            # 修改信息： 发送编号，接收编号
            port_list = list(serial.tools.list_ports.comports())
            # if False:
            if len(port_list) == 0:
                LOGGING.error("No useful COM!")
                exit(-1)
            else:
                Config.ser_obj = serial.Serial(port_list[0].device, 115200)
                fmt = "@BBB"
                read_size = struct.calcsize(fmt)
                while True:
                    read_data = Config.ser_obj.read(read_size)
                    read_data = struct.unpack(fmt, read_data)
                    if read_data != b'':
                        if read_data[0] == 0xAA and read_data[-1] == 0x55:
                            if int(read_data[1]) in Config.RED_CLOR_L:
                                Config.TEAM_COLOR = Config.TEAM_RED
                                Config.CAR_ID = read_data[1]
                                config_ = config("R")
                                # Config.OPCV_DETOR = ArmorDetector(config_, team='R', debug=Config.IMG_INFO)
                                # Config.OPCV_DETOR = ArmorDetector(config_, team='R', debug=False)
                                LOGGING.info("Set current color is RED!")
                                break

                            elif int(read_data[1]) in Config.BLUE_CLOR_L:
                                Config.TEAM_COLOR = Config.TEAM_BLUE
                                Config.CAR_ID = read_data[1]
                                config_ = config("B")
                                # Config.OPCV_DETOR = ArmorDetector(config_, team='B', debug=Config.IMG_INFO)
                                # Config.OPCV_DETOR = ArmorDetector(config_, team='B', debug=False)
                                LOGGING.info("Set current color is BLUE!")
                                break
                            else:
                                LOGGING.error("COM data Robot ID not in color list!")
                        else:
                            LOGGING.error("COM data FLAG data error")
                            time.sleep(0.050)

                    else:
                        LOGGING.info("COM data is NULL")
                        time.sleep(0.2)

        # # 修改信息： 发送编号，接收编号
        # port_list = list(serial.tools.list_ports.comports())
        # # if False:
        # if len(port_list) == 0:
        #     LOGGING.error("No useful COM!")
        #     exit(-1)
        # else:
        #     Config.ser_obj = serial.Serial(port_list[0].device, 115200)
        # # ---debug ----
        # config_ = config("R")
        # Config.OPCV_DETOR = ArmorDetector(config_, team='R', debug=False)

        # CAMERA
        # camera_t = Thread(target=collect_img_main, daemon=True)
        camera_t = Thread(target=video_test_main, daemon=True)

        # DETECTOR
        detect_t = Thread(target=TRTdetector_main, daemon=True)

        # PREDICTOR
        predict_t = Thread(target=KFpredictor_main, daemon=True)
        # 检测先启动，然后是相机线程，防止在检测线程启动过程中队列加入太多图片，防抖
        t_list = [detect_t, camera_t, predict_t]
        # t_list = [detect_t, camera_t]

        for i in t_list:
            i.start()

        #camera_t.join()
        # detect_t.join()
        predict_t.join()
    except:
       traceback.print_exc()
       traceback.print_exc(file=open(log_file_path, "a+"))





