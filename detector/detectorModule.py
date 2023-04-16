# -*- coding:utf-8 -*-
# Data: 2021/11/20 上午12:21

import os
import cv2
import onnx
import onnxruntime
import time
import numpy as np
import traceback

from common.logFile import LOGGING, log_file_path
from common import Config
from common.msgQueue import Q_camera2detector, Q_detector2predictor
from OPENCVdetector.armor_detect_withcarbox import read_morphology_temp, find_contours
from camera.cameraModule import video_test_main

from yolox.data import preproc as preprocess
from yolox.data import VOC_CLASSES
from yolox.data import ValTransform
from detector.yolox.utils import fuse_model, postprocess, vis, demo_postprocess, multiclass_nms
from detector.yolox.models.yolo_head import YOLOXHead

VOC_CLASSES = (
        "car_red",
        "car_blue",
        "car_unknow",
        "watcher_red",
        "watcher_blue",
        "watcher_unknow",
        "armor_red",
        "armor_blue",
        "armor_grey"
)


class YOLODetector(object):
    def __init__(self):
        # load onnx file
        # self.onnx_model = onnx.load_model("yolox_tiny_rm.onnx")
        # self.onnx_model = onnx.load_model("yolox_tiny_200.onnx")
        self.onnx_model = onnx.load_model("yolox_tiny_300.onnx")
        # self.onnx_model = onnx.load_model("your_rm.onnx")
        self.session = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())
        # boundingbox confidence
        self.conf = 0.5
        # 配置己方队伍
        self.team = Config.TEAM_COLOR

    def detect_result(self, input_img):
        input_shape = (416, 416)
        # todo: 测试resize()时间
        origin_img = cv2.resize(input_img, input_shape)
        # YOLOX preprocess
        img, ratio = preprocess(origin_img, input_shape)

        # --------------ONNX FILE PREDICTOR-----------------
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        # YOLOX
        if dets is not None and Config.IMG_INFO:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=VOC_CLASSES)
            # cv2.imwrite("yolo_onn_rst.png", origin_img)
        return dets, origin_img

    def select_target(self, box_list, scores, detect_class):
        # box: [left_up_x, left_up_y, right_down_x, right_down_y]
        # 车
        car_red_list = []
        car_blue_list = []
        car_unknown_list =[]
        # 哨兵
        watcher_red_list = []
        watcher_blue_list = []
        watcher_unknown_list = []
        # 装甲板
        armor_red_list = []
        armor_blue_list = []
        armor_grey_list= []

        select_box_list = {0: car_red_list,
                           1: car_blue_list,
                           2: car_unknown_list,
                           3: watcher_red_list,
                           4: watcher_blue_list,
                           5: watcher_unknown_list,
                           6: armor_red_list,
                           7: armor_blue_list,
                           8: armor_grey_list}
        # armor_cent_list = []
        # car_cent_list = []
        # team_R_target = {"armor": armor_red_list, "car": car_red_list, "watcher": watcher_red_list}
        # team_B_target = {"armor": armor_blue_list, "car": car_blue_list, "watcher":watcher_blue_list}
        # just for DEBUG info
        cls_dic = {0: "car_red",
                    1: "car_blue",
                    2: "car_unknow",
                    3: "watcher_red",
                    4: "watcher_blue",
                    5: "watcher_unknow",
                    6: "armor_red",
                    7: "armor_blue",
                    8: "armor_grey"}
        # conf大于0.5的加入box_list
        for i in range(len(box_list)):
            # box: [left_up_x, left_up_y, right_down_x, right_down_y]
            box = box_list[i]
            cls_id = int(detect_class[i])
            score = scores[i]

            if score < self.conf:
                continue
            select_box_list[cls_id].append(box)

        # -------SELF: BLUE  ENEMY: RED ------
        if self.team == Config.TEAM_BLUE:
            # # 敌方队伍为R
            # # 检测到装甲板
            # # if len(armor_red_list) != 0:
            # if armor_red_list:
            #     # 计算装甲板四个角点坐标,中心和角度解算,根据装甲板box面积大小排序
            #     # changed: 得到装甲板box直接发送预测模块
            #     armor_red_list.sort(key=box_area, reverse=True)
            #
            #
            #     # 计算装甲板四个角点坐标,中心和角度解算
            #     # left_up, left_down, right_up, right_down = point_sort(armor_red_list[0])
            #     # red_armor_cent = box_cent(left_up, left_down, right_up)
            #     # rotation_vector, translation_vector, distance, yaw, pit = gimbal_pit_yaw(left_up, right_up, right_down, left_down, red_armor_cent)
            #     # todo: 世界坐标系计算
            #     # todo: pitch弹道补偿
            #     # todo: 灰色装甲板计数
            #     # todo:防抖，current_shoot_armor new_shoot_armor
            #
            #     # armor_cent_list.append(red_armor_cent)
            #     # armor_angle_list.append([yaw, pit])
            #     # armor_dis_list.append(distance)
            #
            # # 没有装甲板，但是有车，进行跟踪
            # # elif len(car_red_list) != 0:
            # elif car_red_list:
            #     # 计算car box四个角点坐标,中心和角度解算,根据装甲板box面积大小排序
            #     # changed: 得到装甲板box直接发送预测模块
            #     car_red_list.sort(key=box_area, reverse=True)
            #
            #
            #     # 计算car box四个角点坐标,中心和角度解算
            #     # left_up, left_down, right_up, right_down = point_sort(car_red_list[0])
            #     # red_car_cent = box_cent(left_up, left_down, right_up)
            #     # rotation_vector, translation_vector, distance, yaw, pit = gimbal_pit_yaw(left_up, right_up, right_down, left_down, red_car_cent)
            #     # todo: 世界坐标系计算
            #     # car_cent_list.append(red_car_cent)
            #     # car_angle_list.append([yaw, pit])
            #     # car_dis_list.append(distance)
            return armor_red_list, car_red_list
            # todo: 灰色装甲板计数和工程跟踪

        # -------SELF: RED ENEMY: BLUE ------
        if self.team == Config.TEAM_RED:
            # # 敌方队伍为B
            # # 检测到装甲板
            # # if len(armor_blue_list) != 0:
            # if armor_blue_list:
            #     # 根据car box面积进行降序排序，确定击打优先级
            #     armor_blue_list.sort(key=box_area, reverse=True)
            #
            #
            #     # 计算car box中心和角度解算
            #     # left_up, left_down, right_up, right_down = point_sort(armor_blue_list[0])
            #     # blue_armor_cent = box_cent(left_up, left_down, right_up)
            #     # rotation_vector, translation_vector, distance, yaw, pit = gimbal_pit_yaw(left_up, right_up, right_down, left_down, blue_armor_cent)
            #     # todo: 世界坐标系计算
            #     # todo: pitch弹道补偿
            #     # todo: 灰色装甲板计数
            #     # armor_cent_list.append(blue_armor_cent)
            #     # armor_angle_list.append([yaw, pit])
            #     # armor_dis_list.append(distance)
            #     # 完善设计决策以后在直接return
            #     # return pit, yaw, vis_img
            #
            # # 没有装甲板，但是有车，进行跟踪
            # # elif len(car_blue_list) != 0:
            # elif car_blue_list:
            #     # 根据car box面积进行降序排序，确定击打优先级
            #     car_blue_list.sort(key=box_area, reverse=True)
            #
            #
            #     # 计算car box中心和角度解算
            #     # left_up, left_down, right_up, right_down = point_sort(car_blue_list[0])
            #     # blue_car_cent = box_cent(left_up, left_down, right_up)
            #     # rotation_vector, translation_vector, distance, yaw, pit = gimbal_pit_yaw(left_up, right_up, right_down, left_down, blue_car_cent)
            #     # todo: 世界坐标系计算
            #     # car_cent_list.append(blue_car_cent)
            #     # car_angle_list.append([yaw, pit])
            #     # car_dis_list.append(distance)
            #     # 完善设计决策以后在直接return
            #     # return pit, yaw, vis_img
            return armor_blue_list, car_blue_list
            # todo: 灰色装甲板计数和工程跟踪
            # elif
        # return armor_cent_list, car_cent_list


def detector_main():
    try:
        frame_id = 0
        notarget_frame = 0
        # init detect obj
        detect_obj = YOLODetector()
        while True:
        # while not Q_camera2detector.empty():
            try:
                if not Q_camera2detector.empty():
                    detect_start_t = time.time()
                    input_img = Q_camera2detector.get()
                    # ----- DEBUG test -----
                    # cv2.imshow("detect img", input_img)
                    # cv2.waitKey(0)
                    # cv2.imwrite("detectimg", input_img)
                    # get detect result, YOLOX detect box image
                    detect_rst, clas_vis_img = detect_obj.detect_result(input_img)
                    frame_id += 1
                    if detect_rst is None:
                        LOGGING.info("frame id: {frame}, detector rst is NONE, next frame!".format(frame=frame_id))
                        continue
                    # slice final_box, final score, final class index
                    final_boxes, final_scores, final_cls_inds = detect_rst[:, :4], detect_rst[:, 4], detect_rst[:, 5]
                    # 筛选出装甲板列表,车列表,可视化图像
                    armor_list, car_list = detect_obj.select_target(final_boxes, final_scores, final_cls_inds)
                    # 数据发送到预测模块
                    delay_time = time.time() - detect_start_t
                    Q_detector2predictor.put([armor_list, car_list, frame_id, delay_time, clas_vis_img])

                    # if Config.IMG_INFO:
                    #     delt_t = "detect cost: " + str(np.round(delay_time, 3) * 1000) + "ms"
                    #     cv2.putText(clas_vis_img, delt_t, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                    LOGGING.info("Detector put data to predictor, frame: {}".format(frame_id))
                    print("detect frame id: ", frame_id, "detect cost: ", delay_time)

                    # if Config.WRITE_VIDEO:
                    #     video_out.write(angle_vis_img)
                    if Config.IMG_INFO:
                        cv2.imshow("DetectImg", clas_vis_img)
                        cv2.imwrite(str(frame_id)+".png", clas_vis_img)
                        cv2.waitKey(1)

                    if frame_id == Config.TEST_VIDEO_FRAME:
                        LOGGING.info("Detector task done, wait all task done!")
                        time.sleep(1)
                else:
                    notarget_frame += 1
                    if notarget_frame == 300:
                        print("Q_camera2detector empty, waiting camera img!")
                        notarget_frame = 0
                    time.sleep(0.01)
            except:
                if Config.CONSOLE_INFO:
                    traceback.print_exc()
                if Config.LOG_FILE_INFO:
                    traceback.print_exc(log_file_path, "a")
                    continue
    except:
        if Config.CONSOLE_INFO:
            traceback.print_exc()
        if Config.LOG_FILE_INFO:
            traceback.print_exc(log_file_path, "a")


if __name__ == "__main__":
    video_test_main()
    detector_main()
