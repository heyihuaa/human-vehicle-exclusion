import cv2
import numpy as np
import time
from .config import MOTION_SETTING

obj_motion_timer = {}

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)

def judge_vehicle_motion(frame, vehicle_bbox, vehicle_id, prev_gray, thresh):
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    obj_thresh = thresh[y1:y2, x1:x2]
    obj_motion_pixels = np.sum(obj_thresh) / 255
    is_current_moving = obj_motion_pixels > MOTION_SETTING["motion_threshold"] / 10
    current_time = time.time()
    if is_current_moving:
        obj_motion_timer[vehicle_id] = current_time
        motion_color = (255, 0, 0)
        is_moving = True
    else:
        if vehicle_id in obj_motion_timer and current_time - obj_motion_timer[vehicle_id] < MOTION_SETTING["motion_delay"]:
            is_moving = True
            motion_color = (255, 0, 0)
        else:
            obj_motion_timer.pop(vehicle_id, None)
            is_moving = False
            motion_color = (128, 128, 128)
    cv2.putText(frame, f"运动状态：{'运动中' if is_moving else '静止'}",
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 2)
    return is_moving
