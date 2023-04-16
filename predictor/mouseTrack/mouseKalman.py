# -*- coding:utf-8 -*-
# Data: 2021/12/16 下午11:15
'''
kalman2d - 2D Kalman filter using OpenCV
Based on http://jayrambhia.wordpress.com/2012/07/26/kalman-filter/
Copyright (C) 2014 Simon D. Levy
This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with this code. If not, see <http://www.gnu.org/licenses/>.
'''

import cv2

import cv2
import numpy as np
from sys import exit


class Kalman2D(object):
    '''
    A class for 2D Kalman filtering
    '''

    def __init__(self, yaw_speed=0, pit_speed=0, processNoiseCovariance=1e-4, measurementNoiseCovariance=1e-1, errorCovariancePost=0.1):
        '''
        Constructs a new Kalman2D object.
        For explanation of the error covariances see
        http://en.wikipedia.org/wiki/Kalman_filter
        '''
        # 状态空间：位置--2d,速度--2d
        # self.kalman = cv.CreateKalman(4, 2, 0)
        # self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
        # self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        # self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        # 状态空间：位置--2d,速度--2d
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman_state = np.array((4, 1), dtype=np.float) # [x, y, x_speed, y_speed]
        self.kalman_process_noise = np.array((4, 1), dtype=np.float)
        self.kalman_measurement = np.array((2, 1), dtype=np.float)

        # for j in range(4):
        #     for k in range(4):
        #         self.kalman.transitionMatrix[j, k] = 0
        #     self.kalman.transitionMatrix[j, j] = 1
        self.kalman.transitionMatrix = np.mat([[1.0, 0, 1, 0],
                                                [0, 1.0, 0, 1],
                                                [0, 0, 1.0, 0],
                                                [0, 0, 0, 1.0]])
        # 加入速度 x = x + vx, y = y + vy
        # 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1
        # 如果把下面两句注释掉，那么位置跟踪kalman滤波器的状态模型就是没有使用速度信息
        # self.kalman.transitionMatrix[0, 2]=1
        # self.kalman.transitionMatrix[1, 3]=1

        # cv2.SetIdentity(self.kalman.measurement_matrix)
        self.kalman.measurementMatrix = np.eye(2, 1, dtype=np.float)
        # 初始化带尺度的单位矩阵
        # cv2.SetIdentity(self.kalman.process_noise_cov, cv2.RealScalar(processNoiseCovariance))
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]],
                                               np.float32) * processNoiseCovariance  # 系统过程噪声协方差
        # cv2.SetIdentity(self.kalman.measurement_noise_cov, cv2.RealScalar(measurementNoiseCovariance))
        self.kalman.measurementNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]],
                                               np.float32) * measurementNoiseCovariance  # 测量噪声协方差
        # cv2.SetIdentity(self.kalman.error_cov_post, cv2.RealScalar(errorCovariancePost))
        self.kalman.errorCovPost = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]],
                                               np.float32) * errorCovariancePost  # 系统过程噪声协方差

        self.predicted = None
        self.esitmated = None

    def update(self, x, y):
        '''
        Updates the filter with a new X,Y measurement
        '''

        self.kalman_measurement[0] = x
        self.kalman_measurement[1] = y

        self.predicted = self.kalman.predict()
        self.corrected = self.kalman.correct(self.kalman_measurement)

    def getEstimate(self):
        '''
        Returns the current X,Y estimate.
        '''

        return self.corrected[0, 0], self.corrected[1, 0]

    def getPrediction(self):
        '''
        Returns the current X,Y prediction.
        '''

        return self.predicted[0, 0], self.predicted[1, 0]


'''
kalman_mousetracker.py - OpenCV mouse-tracking demo using 2D Kalman filter 
Adapted from
   http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/

Copyright (C) 2014 Simon D. Levy
This code is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU Lesser General Public License
along with this code. If not, see <http://www.gnu.org/licenses/>.
'''

# This delay will affect the Kalman update rate
DELAY_MSEC = 20

# Arbitrary display params
WINDOW_NAME = 'Kalman Mousetracker [ESC to quit]'
WINDOW_SIZE = 500




class MouseInfo(object):
    '''
    A class to store X,Y points
    '''

    def __init__(self):
        self.x, self.y = -1, -1

    def __str__(self):
        return '%4d %4d' % (self.x, self.y)


def mouseCallback(event, x, y, flags, mouse_info):
    '''
    Callback to update a MouseInfo object with new X,Y coordinates
    '''

    mouse_info.x = x
    mouse_info.y = y


def drawCross(img, center, r, g, b):
    '''
    Draws a cross a the specified X,Y coordinates with color RGB
    '''

    d = 5
    t = 2

    color = (r, g, b)

    ctrx = center[0]
    ctry = center[1]

    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t)


def drawLines(img, points, r, g, b):
    '''
    Draws lines
    '''

    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def newImage():
    '''
    Returns a new image
    '''

    return np.zeros((500, 500, 3), np.uint8)


if __name__ == '__main__':

    # Create a new image in a named window
    img = newImage()
    cv2.namedWindow(WINDOW_NAME)

    # Create an X,Y mouse info object and set the window's mouse callback to modify it
    mouse_info = MouseInfo()
    cv2.setMouseCallback(WINDOW_NAME, mouseCallback, mouse_info)

    # Loop until mouse inside window
    while True:

        if mouse_info.x > 0 and mouse_info.y > 0:
            break

        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(1) == 27:
            exit(0)

    # These will get the trajectories for mouse location and Kalman estiamte
    measured_points = []
    kalman_points = []

    # Create a new Kalman2D filter and initialize it with starting mouse location
    kalman2d = Kalman2D()

    # Loop till user hits escape
    while True:

        # Serve up a fresh image
        img = newImage()

        # Grab current mouse position and add it to the trajectory
        measured = (mouse_info.x, mouse_info.y)
        measured_points.append(measured)

        # Update the Kalman filter with the mouse point
        kalman2d.update(mouse_info.x, mouse_info.y)

        # Get the current Kalman estimate and add it to the trajectory
        estimated = [int(c) for c in kalman2d.getEstimate()]
        kalman_points.append(estimated)

        # Display the trajectories and current points
        drawLines(img, kalman_points, 0, 255, 0)
        drawCross(img, estimated, 255, 255, 255)
        drawLines(img, measured_points, 255, 255, 0)
        drawCross(img, measured, 0, 0, 255)

        # Delay for specified interval, quitting on ESC
        cv2.imshow(WINDOW_NAME, img)
        if cv2.waitKey(DELAY_MSEC) == 27:
            break
