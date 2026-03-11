"""
主程序入口
- 视频/摄像头读取：帧获取、预处理
- 模块调度：调用calculator/motion_detector/utils 完成核心逻辑
- 可视化：结果绘制、实时展示
- 报警触发：调用alarmer模块处理报警逻辑
"""
import cv2
import os
import numpy as np
from ultralytics import YOLO
from config import SystemState, STATE_COLOR
from alarmer import trigger_vehicle_person_alarm
from calculator import (
    bbox_bottom_center, compute_homography_matrix, pixel_to_ground,
    calculate_dynamic_braking_distance, mutual_exclusion_model, draw_potato_envelope
)
from motion_detector import SimpleTracker

# ============================================================
# 人车嵌套过滤函数 (可直接放在 core.py，或移到 utils.py)
# ============================================================
def is_fully_contained(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    return (x1_b <= x1_a and y1_b <= y1_a and
            x2_a <= x2_b and y2_a <= y2_b)

def area(box):
    x1, y1, x2, y2 = box
    return max(0, (x2 - x1) * (y2 - y1))

def filter_person_in_forktruck(detections, ratio_thresh=0.4):
    detections = sorted(detections, key=lambda d: area(d['bbox']), reverse=True)
    keep = [True] * len(detections)
    for i, det_a in enumerate(detections):
        if not keep[i]:
            continue
        if det_a['class'] != 'person':
            continue
        box_a = det_a['bbox']
        area_a = area(box_a)
        for j, det_b in enumerate(detections):
            if i == j or not keep[j]:
                continue
            if det_b['class'] != 'fork Truck':
                continue
            box_b = det_b['bbox']
            if is_fully_contained(box_a, box_b):
                area_b = area(box_b)
                if area_b > 0 and area_a / area_b <= ratio_thresh:
                    keep[i] = False
                    break
    return [det for det, flag in zip(detections, keep) if flag]

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    model = YOLO("best.pt")   # 请替换为您的模型路径
    
    video_dir = r"河北12北雨棚\4"
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files.sort()
    
    if not video_files:
        print(f"未在 {video_dir} 找到 mp4 视频文件。")
        video_files = [0]
        
    H = compute_homography_matrix()
    print("[初始化] ISO 3691-4 动态制动包络线 系统准备完毕")

    for video_path in video_files:
        if isinstance(video_path, str):
            print(f"\n[测试] 正在播放视频: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        camera_id = "CAM_01"
        
        person_tracker = SimpleTracker(max_age=15, min_hits=2, iou_threshold=0.3)
        vehicle_tracker = SimpleTracker(max_age=15, min_hits=2, iou_threshold=0.3)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            # 构建检测列表
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                bbox = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                if h < 10 or w < 10:
                    continue

                aspect_ratio = h / w
                det_class = "person" if aspect_ratio >= 1.5 else "fork Truck"
                
                detections.append({
                    "bbox": bbox,
                    "class": det_class,
                    "conf": conf
                })
            
            # 应用嵌套过滤
            filtered_detections = filter_person_in_forktruck(detections, ratio_thresh=0.4)
            
            raw_persons = [d['bbox'] for d in filtered_detections if d['class'] == 'person']
            raw_vehicles = [d['bbox'] for d in filtered_detections if d['class'] == 'fork Truck']

            # 卡尔曼跟踪
            smoothed_persons = person_tracker.update(raw_persons)
            smoothed_vehicles = vehicle_tracker.update(raw_vehicles)

            # 计算每辆车的动态危险区
            vehicle_danger_info = {}
            for v in smoothed_vehicles:
                v_p_pixel = bbox_bottom_center(v["bbox"])
                v_p_real = pixel_to_ground(v_p_pixel, H)

                D_dynamic, extend_p_pixel, v_real = calculate_dynamic_braking_distance(
                    v["vx"], v["vy"], v_p_pixel, H
                )
                
                vehicle_danger_info[v["id"]] = {
                    "p_real": v_p_real,
                    "p_pixel": v_p_pixel,
                    "extend_p_pixel": extend_p_pixel,
                    "D_dynamic": D_dynamic,
                    "v_real": v_real,
                    "bbox": v["bbox"]
                }
                
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), STATE_COLOR[SystemState.SAFE], 2)
                cv2.putText(frame, f"V: {v_real:.1f}m/s", (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 判断人车互斥
            for p in smoothed_persons:
                p_p_pixel = bbox_bottom_center(p["bbox"])
                p_p_real = pixel_to_ground(p_p_pixel, H)
                
                person_state = SystemState.SAFE
                
                for v_id, v_info in vehicle_danger_info.items():
                    d_real, state = mutual_exclusion_model(
                        p_p_real, v_info["p_real"], v_info["D_dynamic"]
                    )
                    
                    if state > person_state:
                        person_state = state
                        
                    if state != SystemState.SAFE:
                        cv2.line(frame, (int(p_p_pixel[0]), int(p_p_pixel[1])), 
                                       (int(v_info["p_pixel"][0]), int(v_info["p_pixel"][1])), 
                                       STATE_COLOR[state], 2)
                        detail = f"人员入侵车辆{v_id}制动区! 距离:{d_real:.1f}m 制动所需:{v_info['D_dynamic']:.1f}m"
                        if state == SystemState.DANGER:
                            trigger_vehicle_person_alarm(camera_id, detail)
                            
                px1, py1, px2, py2 = map(int, p["bbox"])
                cv2.rectangle(frame, (px1, py1), (px2, py2), STATE_COLOR[person_state], 2)
                cv2.circle(frame, (int(p_p_pixel[0]), int(p_p_pixel[1])), 4, (0, 0, 255), -1)

            # 绘制车辆的土豆包络线
            for v_id, v_info in vehicle_danger_info.items():
                max_v_state = SystemState.SAFE
                for p in smoothed_persons:
                    p_p_real = pixel_to_ground(bbox_bottom_center(p["bbox"]), H)
                    _, s = mutual_exclusion_model(p_p_real, v_info["p_real"], v_info["D_dynamic"])
                    if s > max_v_state:
                         max_v_state = s
                         
                draw_potato_envelope(
                    frame, v_info["p_pixel"], v_info["extend_p_pixel"], 
                    v_info["D_dynamic"], max_v_state, H
                )

            # 缩放显示
            h_frame, w_frame = frame.shape[:2]
            if w_frame > 1280:
                 scale = 1280 / w_frame
                 frame = cv2.resize(frame, (int(w_frame * scale), int(h_frame * scale)))

            cv2.imshow("Dynamic Potato Exclusion Zones", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                 print("退出测试。")
                 cap.release()
                 cv2.destroyAllWindows()
                 exit(0)
            elif key == ord("n"):
                 print("切换到下一个视频...")
                 break

        cap.release()
        cv2.destroyAllWindows()