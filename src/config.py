import os
import numpy as np

# ============================================================
# 邮件配置
# ============================================================
EMAIL_SETTING = {
    "smtp_server": "smtp.qq.com",
    "sender": os.getenv("ALARM_EMAIL_SENDER"),
    "auth_code": os.getenv("ALARM_EMAIL_AUTH"),
    "receivers": os.getenv("ALARM_EMAIL_RECEIVERS", "").split(",")
}

# ============================================================
# 报警配置
# ============================================================
ALARM_SETTING = {
    "log_file": "vehicle_person_alarm.log",
    "cool_down": 10
}

LAST_ALARM = {}

# ============================================================
# 测距配置
# ============================================================
DISTANCE_MODE = "homography"
PIXELS_PER_METER = 100.0

# 4 个标定点的像素坐标 (u, v)
CALIB_PIXEL_POINTS = np.float32([
    [1214, 1324],   # 标定点1
    [1780,  922],   # 标定点2
    [1164,  740],   # 标定点3
    [ 631,  962],   # 标定点4
])

# 4 个标定点对应的真实地面坐标 (X, Y)，单位：米
CALIB_REAL_POINTS = np.float32([
    [0.0, 0.0],
    [4.0, 0.0],
    [4.0, 4.0],
    [0.0, 4.0],
])

# ============================================================
# 纯物理制动模型参数 (ISO 3691-4 标准)
# ============================================================
PHYSICS = {
    "FPS": 25.0,             # 视频帧率
    "T_REACTION": 1.0,       # 系统+人的反应时间 (秒)
    "MU": 0.6,               # 摩擦系数
    "G": 9.8,                # 重力加速度 (m/s²)
    "MIN_SAFE_RADIUS": 1.5,  # 静止时的基础安全半径 (米)
    "WARNING_MARGIN": 1.5    # Warning 区域外扩余量 (米)
}

# ============================================================
# 系统状态
# ============================================================
class SystemState:
    SAFE = 0
    WARNING = 1
    DANGER = 2

STATE_COLOR = {
    SystemState.SAFE: (0, 255, 0),      # 绿色
    SystemState.WARNING: (0, 255, 255),  # 黄色
    SystemState.DANGER: (0, 0, 255)      # 红色
}

STATE_TEXT = {
    SystemState.SAFE: "SAFE",
    SystemState.WARNING: "WARNING",
    SystemState.DANGER: "DANGER"
}