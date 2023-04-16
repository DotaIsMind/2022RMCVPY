# -*- coding:utf-8 -*-
# Data: 2021/11/19 下午11:24

import queue

# 后进先出队列
Q_camera2detector = queue.LifoQueue(maxsize=3000)
#Q_camera2detector = queue.Queue(maxsize=3000)
# Q_detector2predictor = queue.LifoQueue(maxsize=3000)
Q_detector2predictor = queue.Queue(maxsize=3000)

