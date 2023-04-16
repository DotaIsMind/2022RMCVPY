# -*- coding:utf-8 -*-
# Data: 2021/11/19 下午11:32

import logging
import time

# 创建一个新的logger别名，不给定则返回的是root logger
LOGGING = logging.getLogger('FileLog')
# 定义日志格式，%(asctime)指定时间格式，%(levelname)指定等级，%(filename)指定文件名，%(lineno)指定输出的行号
strm_log_format = logging.Formatter('%(asctime)s-%(levelname)s-%(filename)s-line:%(lineno)s: %(message)s')
# 设置LOGGING的输出等级
LOGGING.setLevel(logging.DEBUG)

# 实例化StreamHandler类
strm_logger = logging.StreamHandler()

# 以时间命名日志文件
time_fmt = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
log_file_path = time_fmt + '.log'
file_logger = logging.FileHandler(log_file_path)

# # when='D'以天为周期创建日志文件，也可以按照月(M)，小时(H)等
# file_logger = handler.TimedRotatingFileHandler(filename=log_file_name, when='D', encoding='utf-8')
# 设置格式
file_logger.setFormatter(strm_log_format)
# 也可以设置日志文件大小，达到上限后创建新文件
# file_logger = handlers.RotatingFileHandler(filename=log_file_name, maxBytes=1024, encoding='utf-8')
LOGGING.addHandler(file_logger)
LOGGING.addHandler(strm_logger)
