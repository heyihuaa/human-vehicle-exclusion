import cv2
import numpy as np
import time  
from ultralytics import YOLO


model = YOLO("runs/detect/train4/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_frame =None  # 前一帧图像
MOTION_THRESHOLD = 500  # 像素变化阈值（判定运动）
MOTION_DELAY = 5  # 运动状态延迟保持时间（秒）
# 记录每个物体的运动计时 
obj_motion_timer = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if prev_frame is None:
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        continue

    results = model(frame, conf=0.3, device="cpu")
    detections = results[0].boxes

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
    frame_diff = cv2.absdiff(prev_frame, curr_gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    for idx, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = model.names[int(box.cls[0])]
        conf = box.conf[0].item()
        obj_id = f"{cls}_{idx}"  

        obj_thresh = thresh[y1:y2, x1:x2]
        obj_motion_pixels = np.sum(obj_thresh) / 255
        is_current_moving = obj_motion_pixels > MOTION_THRESHOLD/10


        current_time = time.time()  # 当前时间戳
        if is_current_moving:
            # 检测到运动：更新该物体的运动开始时间
            obj_motion_timer[obj_id] = current_time
            is_moving = "运动中（延迟中）"
            color = (0, 0, 255)  # 红色
        else:
            # 未检测到运动：检查是否在5秒延迟期内
            if obj_id in obj_motion_timer:
                elapsed_time = current_time - obj_motion_timer[obj_id]
                if elapsed_time < MOTION_DELAY:
                    # 仍在5秒延迟期内：继续判定为运动
                    is_moving = f"运动中（剩余{int(MOTION_DELAY - elapsed_time)}秒）"
                    color = (0, 0, 255)  # 红色
                else:
                    # 超过5秒：判定为静止，清除计时
                    del obj_motion_timer[obj_id]
                    is_moving = "静止"
                    color = (0, 255, 0)  # 绿色
            else:
                # 从未检测到运动：判定为静止
                is_moving = "静止"
                color = (0, 255, 0)  # 绿色

        # 画框+标注状态
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} {conf:.2f} | {is_moving}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

   
    prev_frame = curr_gray

    #显示画面，按q退出
    cv2.putText(frame, f"运动延迟：{MOTION_DELAY}秒", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("YOLOv11 物体运动检测（5秒延迟）", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()