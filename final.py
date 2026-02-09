import cv2
import math
import time
import json
import os  
import numpy as np  
from datetime import datetime
from ultralytics import YOLO
import yagmail


# 邮件配置
EMAIL_SETTING = {
    "smtp_server": "smtp.qq.com",
    "sender": os.getenv("ALARM_EMAIL_SENDER"),
    "auth_code": os.getenv("ALARM_EMAIL_AUTH"),
    "receivers": os.getenv("ALARM_EMAIL_RECEIVERS", "").split(",")
}


# 报警配置
ALARM_SETTING = {
    "log_file": "vehicle_person_alarm.log",
    "cool_down": 10    
}

LAST_ALARM = {}

# 运动检测配置
MOTION_SETTING = {
    "motion_threshold": 500,  
    "motion_delay": 5,        
    "model_path": "runs/detect/train4/weights/best.pt"  # 这是我自己训练的YOLO模型（用于检测是否可行），如果人车互斥需要改一下
}

# 运动检测状态变量
prev_gray = None  
obj_motion_timer = {}  


# 报警模块
def write_alarm_log(camera_id: str, detail: str, email_ok: bool):
    log_info = {
        "camera_id": camera_id,
        "alarm_type": "人车互斥",
        "alarm_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detail": detail,
        "email_send_success": email_ok
    }
    with open(ALARM_SETTING["log_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(log_info, ensure_ascii=False) + "\n")
    print(f"[日志] 已记录 {camera_id} 告警")

def send_alarm_email(camera_id: str, detail: str) -> bool:
    subject = f"【人车互斥告警】{camera_id} 风险触发"
    content = f"""
告警时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
摄像头编号：{camera_id}
告警详情：{detail}
处理建议：请立即核查现场，避免碰撞事故
—— 大创人车互斥系统
"""
    try:
        client = yagmail.SMTP(
            user=EMAIL_SETTING["sender"],
            password=EMAIL_SETTING["auth_code"],
            host=EMAIL_SETTING["smtp_server"],
            port=465,
            smtp_ssl=True
        )
    except Exception as e:
        print(f"[邮箱错误] 初始化失败：{e}")
        return False

    for i in range(3):
        try:
            client.send(
                to=EMAIL_SETTING["receivers"],
                subject=subject,
                contents=content
            )
            print("[邮箱成功] 告警邮件已发送")
            return True
        except Exception as e:
            print(f"[邮箱重试] 第{i+1}次失败：{e}")
            time.sleep(2)
    return False

def trigger_vehicle_person_alarm(camera_id: str, detail: str):
    now = time.time()
    if camera_id in LAST_ALARM:
        if now - LAST_ALARM[camera_id] < ALARM_SETTING["cool_down"]:
            print(f"[冷却] {camera_id} 冷却中，跳过告警")
            return

    LAST_ALARM[camera_id] = now
    email_ok = send_alarm_email(camera_id, detail)
    write_alarm_log(camera_id, detail, email_ok)


# 互斥模型参数
IMAGE_HEIGHT = 720
BASE_SAFE_RADIUS = 80
SPEED_FACTOR = 30


# 系统状态
class SystemState:
    SAFE = 0
    WARNING = 1
    DANGER = 2

STATE_COLOR = {
    SystemState.SAFE: (0, 255, 0),
    SystemState.WARNING: (0, 255, 255),
    SystemState.DANGER: (0, 0, 255)
}


# 工具函数
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ( (x1 + x2) / 2, (y1 + y2) / 2 )

def pixel_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def perspective_weight(y):
    return 1.2 * (y / IMAGE_HEIGHT) + 0.5

def mutual_exclusion_model(person, vehicle):
    px, py = bbox_center(person["bbox"])
    vx, vy = bbox_center(vehicle["bbox"])

    d_pixel = pixel_distance((px, py), (vx, vy))
    d_real = d_pixel * perspective_weight(max(py, vy))

    safe_radius = BASE_SAFE_RADIUS + SPEED_FACTOR

    if d_real < safe_radius:
        return d_real, safe_radius, SystemState.DANGER
    elif d_real < safe_radius * 1.2:
        return d_real, safe_radius, SystemState.WARNING
    else:
        return d_real, safe_radius, SystemState.SAFE

# 运动检测
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)

def judge_vehicle_motion(frame, vehicle_bbox, vehicle_id, curr_gray, thresh):
    global obj_motion_timer
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    # 计算车辆区域内的运动像素
    obj_thresh = thresh[y1:y2, x1:x2]
    obj_motion_pixels = np.sum(obj_thresh) / 255
    is_current_moving = obj_motion_pixels > MOTION_SETTING["motion_threshold"] / 10
    
    current_time = time.time()
    # 判断最终运动状态
    if is_current_moving:
        obj_motion_timer[vehicle_id] = current_time
        is_moving = "运动中（延迟中）"
        motion_color = (255, 0, 0)  # 蓝色__运动
    else:
        if vehicle_id in obj_motion_timer:
            elapsed_time = current_time - obj_motion_timer[vehicle_id]
            if elapsed_time < MOTION_SETTING["motion_delay"]:
                is_moving = f"运动中（剩余{int(MOTION_SETTING['motion_delay'] - elapsed_time)}秒）"
                motion_color = (255, 0, 0)
            else:
                del obj_motion_timer[vehicle_id]
                is_moving = "静止"
                motion_color = (128, 128, 128)  # 灰色__静止
        else:
            is_moving = "静止"
            motion_color = (128, 128, 128)
    
    # 在车辆框下方标注运动状态
    cv2.putText(
        frame,
        f"运动状态：{is_moving}",
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        motion_color,
        2
    )
    return False if is_moving == "静止" else True


# 主程序
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    motion_model = YOLO(MOTION_SETTING["model_path"])
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera_id = "CAM_01"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        persons = []
        vehicles = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            bbox = box.xyxy[0].tolist()

            if cls_name == "person":
                persons.append({"bbox": bbox})
            elif cls_name in ["truck", "car", "bus"]: 
                vehicles.append({"bbox": bbox})

        # 车辆运动检测逻辑
        if prev_gray is None:
            prev_gray = preprocess_frame(frame)
        else:
            curr_gray = preprocess_frame(frame)
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # 遍历车辆判断运动状态
            for j, v in enumerate(vehicles):
                vehicle_id = f"vehicle_{j}"
                is_vehicle_moving = judge_vehicle_motion(frame, v["bbox"], vehicle_id, curr_gray, thresh)
            
            prev_gray = curr_gray

        # 人车互斥判断和可视化
        for i, p in enumerate(persons):
            for j, v in enumerate(vehicles):
                distance, safe_radius, state = mutual_exclusion_model(p, v)

                # 报警触发 
                if state == SystemState.DANGER:
                    detail = (
                        f"人员{i} 与车辆{j} 距离 {distance:.2f}，"
                        f"低于安全阈值 {safe_radius:.2f}"
                    )
                    trigger_vehicle_person_alarm(camera_id, detail)

                # 可视化 
                px1, py1, px2, py2 = map(int, p["bbox"])
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])

                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), STATE_COLOR[state], 2)

                cv2.putText(
                    frame,
                    ["SAFE", "WARNING", "DANGER"][state],
                    (vx1, vy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    STATE_COLOR[state],
                    2
                )

        # 添加运动检测配置标注
        cv2.putText(
            frame,
            f"运动延迟：{MOTION_SETTING['motion_delay']}秒",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow("Human-Vehicle Mutual Exclusion System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()