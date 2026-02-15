import cv2
import json
from ultralytics import YOLO
import paho.mqtt.client as mqtt
from .calculator import compute_homography_matrix, mutual_exclusion_model
from .motion_detector import preprocess_frame, judge_vehicle_motion
from .alarmer import trigger_vehicle_person_alarm
from .config import DISTANCE_MODE, ALARM_DISTANCE_M, STATE_COLOR, STATE_TEXT
from .config import MQTT_BROKER, MQTT_PORT, DEVICE_ID, MQTT_TOPIC

def main():
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print("[MQTT] 已连接到 Broker")

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera_id = "CAM_01"
    prev_gray = None
    H = compute_homography_matrix() if DISTANCE_MODE=="homography" else None

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame)[0]
        persons, vehicles = [], []
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            bbox = box.xyxy[0].tolist()
            if cls_name=="person":
                persons.append({"bbox": bbox})
            elif cls_name in ["truck","car","bus"]:
                vehicles.append({"bbox": bbox})

        # 运动检测
        if prev_gray is None:
            prev_gray = preprocess_frame(frame)
        else:
            curr_gray = preprocess_frame(frame)
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            for j, v in enumerate(vehicles):
                vehicle_id = f"vehicle_{j}"
                judge_vehicle_motion(frame, v["bbox"], vehicle_id, prev_gray, thresh)
            prev_gray = curr_gray

        # 人车互斥判断 + 报警 + MQTT
        for i, p in enumerate(persons):
            for j, v in enumerate(vehicles):
                d_real, state, pt_person, pt_vehicle = mutual_exclusion_model(p["bbox"], v["bbox"], H)
                if state == 2:  # DANGER
                    detail = f"人员{i} 与车辆{j} 实际距离 {d_real:.2f}m，低于安全阈值 {ALARM_DISTANCE_M}m"
                    trigger_vehicle_person_alarm(camera_id, detail)
                    payload = {"device_id": DEVICE_ID, "alarm": 1, "driver_present": 1, "outer_intrusion": 1,
                               "timestamp": json.dumps(d_real)}
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                    print("[MQTT] 已发布报警")

                # 可视化
                px1, py1, px2, py2 = map(int, p["bbox"])
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)
                cv2.putText(frame, f"Person {i}", (px1, py1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                color = STATE_COLOR[state]
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2)
                cv2.putText(frame, STATE_TEXT[state], (vx1, vy1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"{d_real:.2f}m", (vx1, vy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pt_p = (int(pt_person[0]), int(pt_person[1]))
                pt_v = (int(pt_vehicle[0]), int(pt_vehicle[1]))
                cv2.line(frame, pt_p, pt_v, color, 2)
                mid_x, mid_y = (pt_p[0]+pt_v[0])//2, (pt_p[1]+pt_v[1])//2
                cv2.putText(frame, f"{d_real:.2f}m", (mid_x+5, mid_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"Mode: {DISTANCE_MODE} | Alarm: {ALARM_DISTANCE_M}m", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.imshow("Human-Vehicle Mutual Exclusion System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
