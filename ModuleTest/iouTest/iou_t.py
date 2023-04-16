# -*- coding:utf-8 -*-
# Data: 2022/2/23 下午9:28

import numpy as np


# def get_IOU(left_up, right_down, grey_left_up, grey_right_down):
#     # ----
#     # 已经失效
#     # ----
#     # 两个box左上坐标x, y的较大值 * 两个box右下坐标x, y的较小值 = 交集的面积
#     intersect_box = [] # 左上x, 左上y, 右下x， 右下y
#     # a与b交集左上x
#     if left_up[0] > grey_left_up[0]:
#         intersect_box.append(left_up[0])
#     else:
#         intersect_box.append(grey_left_up[0])
#     # a与b交集左上y
#     if left_up[1] > grey_left_up[1]:
#         intersect_box.append(left_up[1])
#     else:
#         intersect_box.append(grey_left_up[1])
#     # a与b交集右下x
#     if right_down[0] < grey_right_down[0]:
#         intersect_box.append(right_down[0])
#     else:
#         intersect_box.append(grey_right_down[0])
#     # a与b交集右下y
#     if right_down[1] < grey_right_down[1]:
#         intersect_box.append(right_down[1])
#     else:
#         intersect_box.append(grey_right_down[1])
#
#     area_intersect = (intersect_box[2] - intersect_box[0]) * (intersect_box[3] - intersect_box[1])
#
#     a_box = np.array(right_down) - np.array(left_up)
#     area_a = a_box[0] * a_box[1]
#
#     b_box = np.array(grey_right_down) - np.array(grey_left_up)
#     area_b = b_box[0] * b_box[1]
#
#     IOU = np.around(area_intersect / (area_a + area_b - area_intersect), 2)
#     print("IOU: ", IOU)
#     if IOU > 0.5:
#         return True


def bb_intersection_over_union(left_up, right_down, grey_left_up, grey_right_down):
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


if __name__ == "__main__":
    left_up = [1, 1]
    right_down = [2, 2]
    grey_left_up = [7, 7]
    grey_right_down = [8, 8]
    # rst = get_IOU(left_up, right_down, grey_left_up, grey_right_down)
    rst = bb_intersection_over_union(left_up, right_down, grey_left_up, grey_right_down)