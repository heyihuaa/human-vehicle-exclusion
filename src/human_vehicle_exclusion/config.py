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

# ============================================================
# 测距配置
# ============================================================
DISTANCE_MODE = "scale"  # "scale" 或 "homography"

PIXELS_PER_METER = 100.0  # 比例尺模式

CALIB_PIXEL_POINTS = np.float32([
    [100, 100],
    [540, 100],
    [100, 380],
    [540, 380],
])

CALIB_REAL_POINTS = np.float32([
    [0.0, 0.0],
    [5.0, 0.0],
    [0.0, 3.0],
    [5.0, 3.0],
])

ALARM_DISTANCE_M = 1.5
WARNING_DISTANCE_M = 3.0

# ============================================================
# MQTT 配置
# ============================================================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
DEVICE_ID = "FORK-001"
MQTT_TOPIC = f"factory/forklift/{DEVICE_ID}/alarm"
