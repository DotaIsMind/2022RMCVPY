# -*- coding:utf-8 -*-
# Data: 2021/12/9 下午10:31

import cv2
import torch
import time
import traceback
import numpy as np
from torch2trt import TRTModule

from yolox.data.datasets.voc_classes import VOC_CLASSES
from yolox.data import ValTransform
from yolox.utils import vis, demo_postprocess, multiclass_nms
from yolox.models.yolo_head import YOLOXHead

from common import Config
from common.msgQueue import Q_camera2detector, Q_detector2predictor
from common.logFile import LOGGING, log_file_path
# from predictor.predictorModule import KFpredictor_main
# from OPENCVdetector.armor_detector import opcv_detector


class YOLOXDetectorTRT(object):
    def __init__(
        self,
        trt_file=None,
        fp16=False,
        legacy=False,
        decode=None,
        cls_names=VOC_CLASSES,
    ):
        self.cls_names = cls_names
        self.num_classes = 9   # 一共9类
        self.conf = 0.3  # confidence score
        self.nmsthre = 0.45  # nms value
        self.test_size = (416, 416)
        self.device = "gpu"
        self.decode = decode
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.team = Config.TEAM_COLOR

        if trt_file is not None:

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            # x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            # x = torch.ones(1, 3, self.test_size[0], self.test_size[1]).cuda()
            # self.model(x)
            self.model = model_trt

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        # output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        # scores = output[:, 4] * output[:, 5]
        scores = output[:, 4]
        cls = output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def inference(self, img):
        '''
        @brief: 获取检测结果
        @param: 待检测图像
        @return: 检测结果可视化图像
        '''
        # 相机模块进行了resize
        # input_shape = (416, 416)
        # img = cv2.resize(img, input_shape)
        img_info = {}
        vis_img = img
        img_info["raw_img"] = img
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        # img_pre_start = time.time()
        img, _ = self.preproc(img, None, self.test_size)
        #print("detector size:", img.shape)
        # print("DETOR: img shape: {}, preproc cost: {}".format(img.shape, time.time() - img_pre_start)*1000)
        img = torch.from_numpy(img).unsqueeze(0)
        # img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = outputs.cpu().numpy()
            outputs = demo_postprocess(outputs[0], self.test_size)
        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

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
            vis_img = vis(vis_img, final_boxes, final_scores, final_cls_inds,
                             conf=0.3, class_names=VOC_CLASSES)
            # vis_img = self.visual(dets, img_info, self.conf)
            # vis_img = self.visual(dets, img_info)

        # return dets, vis_img
        return dets, vis_img

    def select_target(self, box_list, scores, detect_class):
        '''
        @brief: 根据颜色和检测结果选择敌方目标
        @param: 检测结果，置信度，检测类别
        @return: 敌方box列表
        '''
        # box: [left_up_x, left_up_y, right_down_x, right_down_y]
        # 车
        car_red_list = []
        car_blue_list = []
        car_unknown_list = []
        # 哨兵
        watcher_red_list = []
        watcher_blue_list = []
        watcher_unknown_list = []
        # 装甲板
        armor_red_list = []
        armor_blue_list = []
        armor_grey_list = []

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
        # ---- DEBUG info ----
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
            # 敌方队伍为R
            return armor_red_list, car_red_list, armor_grey_list
            # todo: 灰色装甲板计数和工程跟踪

        # -------SELF: RED ENEMY: BLUE ------
        if self.team == Config.TEAM_RED:
            # 敌方队伍为B
            return armor_blue_list, car_blue_list, armor_grey_list
            # todo: 灰色装甲板计数和工程跟踪


def trt_image_demo(predictor, input_image):
    start_t = time.time()
    outputs, vis_img = predictor.inference(input_image)
    print(time.time()- start_t)
    if Config.WRITE_VIDEO:
        cv2.imwrite("trt_rst.png", vis_img)
    # ch = cv2.waitKey(0)
    # if ch == 27 or ch == ord("q") or ch == ord("Q"):
    return outputs, vis_img


def TRTdetector_main():
    try:
        frame_id = 0
        no_target_frame = 0
        # init detect obj
        yolox_head_obj = YOLOXHead(num_classes=9)
        # trt_engine_file = "rm_model_trt_1060.pth"
        # trt_engine_file = "300_epoch_1060.pth"
        # trt_engine_file = "nx20w_model_trt.pth"
        #trt_engine_file = "nx_model_trt.pth"
        trt_engine_file = "model_trt.pth"

        last_frame_id = 0

        car_list = []
        grey_armor_list = []

        predictor = YOLOXDetectorTRT(trt_file=trt_engine_file, decode=yolox_head_obj)
        while True:
            if not Q_camera2detector.empty():
                detect_start_t = time.time()
                camera_msg = Q_camera2detector.get()
                input_img = camera_msg[0]
                img_id = camera_msg[1]
                cam_start_t = camera_msg[2]
                if frame_id > img_id:
                    print("CAM2DETECT queue size:{}".format(Q_detector2predictor.qsize()))
                    Q_camera2detector.get()
                    continue
             
                # ----- DEBUG test -----
                # cv2.imshow("detect img", input_img)
                # cv2.waitKey(1)
                # cv2.imwrite("detectimg", input_img)

                # get detect result, YOLOX detect box image
                if Config.TRT_FLAG:
                    detect_rst, vis_img = predictor.inference(input_img)
                    frame_id = img_id
                    if detect_rst is None:
                        no_target_frame += 1
                        if no_target_frame == 1000:
                            LOGGING.info("DETOR: frame id: {frame}, detector rst is NONE, next frame!".format(frame=frame_id))
                            no_target_frame = 0
                        continue
                    # slice final_box, final score, final class index
                    final_boxes, final_scores, final_cls_inds = detect_rst[:, :4], detect_rst[:, 4], detect_rst[:, 5]
                    # 筛选出装甲板列表,车列表,可视化图像
                    armor_list, car_list, grey_armor_list = predictor.select_target(final_boxes, final_scores, final_cls_inds)

                # armor_list = []
                # -------- 如果YOLOX没有检测结果，调用OPENCV -------
                # if len(armor_list) == 0:
                if Config.OPCV_FLAG:
                    frame_id = img_id
                    armor_list = Config.OPCV_DETOR.detect(input_img)
                    if armor_list:
                        # armor_list, vis_img = opcv_detector(armor_list, input_img)
                        pass
                if armor_list:
                    # 数据发送到预测模块
                    delay_time = int((time.time() - cam_start_t) * 1000)
                    Q_detector2predictor.put([armor_list, car_list, grey_armor_list, frame_id, delay_time, vis_img])
                else:
                    continue

                # LOGGING.info("DETOR: Detector put data to predictor, frame: {}, detect cost: {}".format(frame_id, (
                #             time.time() - detect_start_t) * 1000))

                # 控制输出不要太多
                if frame_id - last_frame_id > 20:
                    LOGGING.info("DETOR: Detector put data to predictor, frame: {}, detect cost: {}, all cost:{}".format(frame_id, (time.time()-detect_start_t)*1000, delay_time))
                    last_frame_id = frame_id
                if Config.IMG_INFO:
                    cv2.imshow("DetectImg", vis_img)
                    cv2.waitKey(1)
                    # cv2.imwrite("TRT_rst.png", vis_img)
                #dete_rst = [armor_list, car_list, grey_armor_list, frame_id, delay_time, vis_img]
                #KFpredictor_main(dete_rst)

                if Config.TEST_VIDEO_FRAME:
                    if frame_id == Config.TEST_VIDEO_FRAME:
                        LOGGING.info("DETOR: Detector task done, wait all task done!")
                        # time.sleep(1)
                        return
            else:
                no_target_frame += 1
                if no_target_frame == 1000:
                    LOGGING.info("DETOR: no target frame num 1000, waiting target frame!")
                    no_target_frame = 0
                time.sleep(0.010)
    except:
        if Config.CONSOLE_INFO:
            traceback.print_exc()
        if Config.LOG_FILE_INFO:
            traceback.print_exc(file=open(log_file_path, "a+"))


if __name__ == "__main__":
    # yolox_head_obj = YOLOXHead(num_classes=9)
    # trt_enging_file = "yolox_tiny_300_trt.pth"
    #
    # predictor = YOLOXDetectorTRT(trt_file=trt_enging_file, decode=yolox_head_obj)
    input_img = cv2.imread("/home/dota/RMCV/20221YNUrmcv_main/ModuleTest/detectTest/opencv_test.png")

    input_img = cv2.resize(input_img, (416, 416))

    # input_img = cv2.resize(input_img, (416, 416))
    Q_camera2detector.put([input_img, 1, time.time()])
    # trt_image_demo(predictor, input_img)
    # video_test_main()
    TRTdetector_main()
