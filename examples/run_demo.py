import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import paho.mqtt.client as mqtt

from human_vehicle_exclusion.config import *
from human_vehicle_exclusion.mutual_exclusion import mutual_exclusion_model
from human_vehicle_exclusion.alarmer import trigger_vehicle_person_alarm
from human_vehicle_exclusion.visualizer import draw_frame
from human_vehicle_exclusion.distance import compute_homography_matrix
from human_vehicle_exclusion.system_state import SystemState

# MQTT
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# YOLO 模型
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera_id = "CAM_01"

H = compute_homography_matrix() if DISTANCE_MODE=="homography" else None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    persons, vehicles = [], []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        bbox = box.xyxy[0].tolist()
        if cls_name=="person":
            persons.append({"bbox": bbox})
        elif cls_name in ["truck","car","bus"]:
            vehicles.append({"bbox": bbox})

    for i,p in enumerate(persons):
        for j,v in enumerate(vehicles):
            d_real, state, pt_person, pt_vehicle = mutual_exclusion_model(p["bbox"],v["bbox"],H)
            if state==SystemState.DANGER:
                detail = f"人员{i} 与车辆{j} 距离 {d_real:.2f}m < {ALARM_DISTANCE_M}m"
                trigger_vehicle_person_alarm(camera_id, detail)
                mqtt_payload = {
                    "device_id": DEVICE_ID,
                    "alarm":1,
                    "driver_present":1,
                    "outer_intrusion":1,
                    "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                mqtt_client.publish(MQTT_TOPIC,json.dumps(mqtt_payload),qos=1)

            draw_frame(frame, p, v, state, d_real, pt_person, pt_vehicle, i, j)

    cv2.imshow("Human-Vehicle Exclusion System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
