# -*- coding:utf-8 -*-
# Data: 2022/5/4 下午1:12
import numpy as np


# class CAR_1:
#     pass
#
#
# class CAR_3:


# spin infantry
class CAR_INFANTRY:
    # ------ 舵轮相机:CAR_ID = 4 ------
    infantry4_focal_length = np.array([1.281890057808269, 1.276213926816939]) * 1000
    img_x_resize_factor = 416 / infantry4_focal_length[0]
    img_y_resize_factor = 416 / 1024
    infantry4_instrimat = np.array(
        [[1290.5269040172795485 * img_x_resize_factor, 0.0000000000000000, 643.0843358495335451 * img_x_resize_factor],
         [0.0000000000000000, 1286.5137262745745375 * img_y_resize_factor, 650.8023899252162892 * img_y_resize_factor],
         [0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    # infantry4_dist_coeffs = np.transpose(
    #     [-0.208454109225372, 0.040445789016608, 0.380233252688816, 0.0000000000000000, 0.0000000000000000])
    infantry4_dist_coeffs = np.array(
        [-0.208454109225372, 0.040445789016608, 0.380233252688816, 0.0000000000000000, 0.0000000000000000])

class CAR_GUARD:
    # ------ 舵轮相机:CAR_ID = 4 ------
    infantry4_focal_length = np.array([1.281890057808269, 1.276213926816939]) * 1000
    img_x_resize_factor = 416 / infantry4_focal_length[0]
    img_y_resize_factor = 416 / 1024
    infantry4_instrimat = np.array(
        [[1290.5269040172795485 * img_x_resize_factor, 0.0000000000000000, 643.0843358495335451 * img_x_resize_factor],
         [0.0000000000000000, 1286.5137262745745375 * img_y_resize_factor, 425.8023899252162892 * img_y_resize_factor],
         [0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
    # infantry4_dist_coeffs = np.transpose(
    #     [-0.208454109225372, 0.040445789016608, 0.380233252688816, 0.0000000000000000, 0.0000000000000000])
    infantry4_dist_coeffs = np.array(
        [-0.208454109225372, 0.040445789016608, 0.380233252688816, 0.0000000000000000, 0.0000000000000000])

# class CAR_5:
#     pass
#
#
# class CAR_7:
#     pass


# 相机标定参数文件
CameraCalibParamDic = {103: CAR_INFANTRY, 3: CAR_INFANTRY, 4:CAR_INFANTRY, 104: CAR_INFANTRY, 107:CAR_GUARD, 7:CAR_GUARD}






