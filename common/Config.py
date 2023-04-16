# -*- coding:utf-8 -*-
# Data: 2021/11/22 上午12:25
"""
common config

"""

# ---- SELF TEAM CONFIG ----
# ---- 0: RED 1: BLUE ----

TEAM_RED = 0
TEAM_BLUE = 1

RED_CLOR_L = [1, 3, 4, 5, 7]
BLUE_CLOR_L = [101, 103, 104, 105, 107]
TEAM_COLOR = TEAM_RED  # RED：0   BLUE: 1
CAR_ID = 103

# ---- CAMERA CONFIG ----
FPS = 100
IMAGE_SHAPE = (1024, 1080)

# ---- GOLOBAL CONFIG ----
CVMSG2COM = False
ser_obj = None # 串口对象
OPCV_DETOR = None
TRT_FLAG = True
OPCV_FLAG = False


# ---- DEBUG CONFIG ----
CONSOLE_INFO = True    # 在控制台输出Log
LOG_FILE_INFO = True   # Log写入文件/
IMG_INFO = True        # 调试时实时显示IMG,但是多线程下OpenCV的imshow()可能不管用
DEBUG_INFO = True
WRITE_VIDEO = False # 将检测数据和预测数据写入视频

TEST_VIDEO_FRAME = None # 视频测试时控制测试时长
