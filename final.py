import cv2
import math
import time
import json
from datetime import datetime
from ultralytics import YOLO
import yagmail


# 邮件配置

EMAIL_SETTING = {
    "smtp_server": "smtp.qq.com",
    "sender": "3403489687@qq.com",
    "auth_code": "ibiqexfhuglidbdd",   
    "receivers": ["yuz11_4781@qq.com"]
}


# 报警配置

ALARM_SETTING = {
    "log_file": "vehicle_person_alarm.log",
    "cool_down": 10    
}

LAST_ALARM = {}


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


# 主程序

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
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

        cv2.imshow("Human-Vehicle Mutual Exclusion System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# "C:\Program Files\Python311\python.exe" final.py