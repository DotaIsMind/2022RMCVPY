#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import onnx

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from detector.yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


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

if __name__ == '__main__':
    # args = make_parser().parse_args()

    # input_shape = tuple(map(int, args.input_shape.split(',')))
    # origin_img = cv2.imread(args.image_path)

    input_shape = (416, 416)
    input_img = cv2.imread("opencv_test.png")
    origin_img = cv2.resize(input_img, input_shape)
    img, ratio = preprocess(origin_img, input_shape)

    # session = onnxruntime.InferenceSession(args.model)
    # add by FT
    onnx_model = onnx.load_model("your_yolox.onnx")
    session =onnxruntime.InferenceSession(onnx_model.SerializeToString())

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    # predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
        #                  conf=args.score_thr, class_names=COCO_CLASSES)
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=0.3, class_names=VOC_CLASSES)

    # mkdir(args.output_dir)
    # output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    # cv2.imwrite(output_path, origin_img)
    cv2.imwrite("onn_ing.png", origin_img)


