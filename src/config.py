import numpy as np

# ===========================
# 邮件配置
# ===========================
EMAIL_SETTING = {
    "smtp_server": "smtp.qq.com",
    "sender": "your_email@qq.com",
    "auth_code": "your_auth_code",
    "receivers": ["receiver1@qq.com", "receiver2@qq.com"]
}

# ===========================
# 报警配置
# ===========================
ALARM_SETTING = {
    "log_file": "vehicle_person_alarm.log",
    "cool_down": 10
}

# ===========================
# 测距配置
# ===========================
DISTANCE_MODE = "scale"  # "scale" 或 "homography"
PIXELS_PER_METER = 100.0
ALARM_DISTANCE_M = 1.5
WARNING_DISTANCE_M = 3.0

CALIB_PIXEL_POINTS = np.float32([[100, 100], [540, 100], [100, 380], [540, 380]])
CALIB_REAL_POINTS = np.float32([[0.0, 0.0], [5.0, 0.0], [0.0, 3.0], [5.0, 3.0]])

# ===========================
# 车辆运动检测配置
# ===========================
MOTION_SETTING = {
    "motion_threshold": 500,
    "motion_delay": 5
}

# ===========================
# MQTT 配置
# ===========================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
DEVICE_ID = "FORK-001"
MQTT_TOPIC = f"factory/forklift/{DEVICE_ID}/alarm"

# ===========================
# 系统状态
# ===========================
class SystemState:
    SAFE = 0
    WARNING = 1
    DANGER = 2

STATE_COLOR = {
    SystemState.SAFE: (0, 255, 0),
    SystemState.WARNING: (0, 255, 255),
    SystemState.DANGER: (0, 0, 255)
}

STATE_TEXT = {
    SystemState.SAFE: "SAFE",
    SystemState.WARNING: "WARNING",
    SystemState.DANGER: "DANGER"
}
