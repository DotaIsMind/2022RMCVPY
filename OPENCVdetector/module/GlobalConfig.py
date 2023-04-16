# ----------------------------------------------
#
#           SCURM_Vision_Global_Config
#               Coding By Pikachu
#                  全局配置信息
#
#            LAST_UPDATE:OCT30/2020
#
# ----------------------------------------------
global_debug_flag = True
camera_resol_high = 416
camera_resol_widt = 416
camera_resol_sfps = 98   # 设定帧率
camera_resol_tbgr = False

'''
-----------------
装甲板识别配置部分
'''

# 装甲板识别配置
armor_detect_config = {
    # 形态学核阈值
    "open": 1,
    "close": 12,
    "erode": 3,
    "dilate": 9,

    # 装甲板筛选参数
    "area_threshold": 40,  # 面积
    "w_h_ratio": 1.4,  # 灯条长宽比例阈值
    "diff_x": 1.5,
    "diff_y": 1.0,
    "diff_h": 0.2
}

# 敌队为RED时HSV阈值
# TODO: 需要临场测试数值
enemy_red_threshold = {
    "hmin": 15,
    "hmax": 213,
    "smin": 3,
    "smax": 66,
    "vmin": 224,
    "vmax": 255,
}

# 敌人为BLUE时HSV阈值
# TODO: 需要临场测试数值
enemy_blue_threshold = {
    "hmin": 15,
    "hmax": 213,
    "smin": 3,
    "smax": 66,
    "vmin": 224,
    "vmax": 255,
}

'''
-----------------
'''