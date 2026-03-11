import cv2
import math
import time
import json
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import yagmail
from scipy.optimize import linear_sum_assignment


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

# 4 个标定点的像素坐标 (u, v) (已通过标定工具更新)
CALIB_PIXEL_POINTS = np.float32([
    [1214, 1324],   # 标定点1 像素坐标
    [1780,  922],   # 标定点2 像素坐标
    [1164,  740],   # 标定点3 像素坐标
    [ 631,  962],   # 标定点4 像素坐标
])

# 4 个标定点对应的真实地面坐标 (X, Y)，单位：米
CALIB_REAL_POINTS = np.float32([
    [0.0, 0.0],   # 标定点1 真实坐标
    [4.0, 0.0],   # 标定点2 真实坐标
    [4.0, 4.0],   # 标定点3 真实坐标
    [0.0, 4.0],   # 标定点4 真实坐标
])


# ============================================================
# 纯物理制动模型参数 (ISO 3691-4 标准)
# ============================================================
PHYSICS = {
    "FPS": 25.0,             # 视频帧率，用于将像素每帧速度转为秒级速度
    "T_REACTION": 1.0,       # 系统+人的反应时间缓冲 (秒)
    "MU": 0.6,               # 典型车间地坪摩擦系数
    "G": 9.8,                # 重力加速度 (m/s^2)
    "MIN_SAFE_RADIUS": 1.5,  # 车辆完全静止时的极限基础安全半径 (米)
    "WARNING_MARGIN": 1.5    # Warning 状态比 Danger 状态外扩的安全余量 (米)
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


# ============================================================
# 报警模块
# ============================================================

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
    if not EMAIL_SETTING["sender"] or not EMAIL_SETTING["auth_code"]:
         return False
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
            time.sleep(2)
    return False

def trigger_vehicle_person_alarm(camera_id: str, detail: str):
    now = time.time()
    if camera_id in LAST_ALARM:
        if now - LAST_ALARM[camera_id] < ALARM_SETTING["cool_down"]:
            return

    LAST_ALARM[camera_id] = now
    email_ok = send_alarm_email(camera_id, detail)
    write_alarm_log(camera_id, detail, email_ok)


# ============================================================
# 卡尔曼滤波与目标追踪模块
# ============================================================

def bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    a = w / float(h) if h > 0 else 0
    return np.array([cx, cy, a, h]).reshape(4, 1)

def z_to_bbox(z):
    cx, cy, a, h = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
    w = a * h
    return [cx - w/2., cy - h/2., cx + w/2., cy + h/2.]

def calculate_iou(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

class BBoxKalmanFilter:
    def __init__(self, dt=1.0):
        self.ndim = 4
        self.dt = dt
        self.F = np.eye(2 * self.ndim)
        for i in range(self.ndim):
            self.F[i, i + self.ndim] = self.dt
            
        self.H = np.eye(self.ndim, 2 * self.ndim)
        
        self.P = np.eye(2 * self.ndim) * 10.0
        for i in range(self.ndim, 2 * self.ndim):
            self.P[i, i] *= 1000.0 
            
        self.Q = np.eye(2 * self.ndim)
        # 过程噪声越小，平滑作用越强，但跟随越迟钝。减小 cx/cy 过程噪声：
        self.Q[0:self.ndim, 0:self.ndim] *= 0.005 
        # vx/vy 过程噪声减小，防止速度来回跳变：
        self.Q[self.ndim:, self.ndim:] *= 0.00001
        
        # 测量噪声 R 越大，越不信任当前 YOLO 传来的高频抖动框
        self.R = np.eye(self.ndim) * 1.0
        self.X = np.zeros((2 * self.ndim, 1))

    def initiate(self, measurement):
        self.X[:self.ndim] = measurement
        self.X[self.ndim:] = 0

    def predict(self):
        self.X = np.dot(self.F, self.X)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement):
        Y = measurement - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = self.X + np.dot(K, Y)
        I = np.eye(2 * self.ndim)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
    def get_state_bbox(self):
        return z_to_bbox(self.X[:self.ndim])
        
    def get_velocity(self):
        """获取 cx 和 cy 在像素系下的速度分量 (v_x, v_y)"""
        return self.X[4, 0], self.X[5, 0]

class TrackedObject:
    _id_count = 0
    def __init__(self, bbox):
        TrackedObject._id_count += 1
        self.id = TrackedObject._id_count
        self.kf = BBoxKalmanFilter()
        self.kf.initiate(bbox_to_z(bbox))
        self.time_since_update = 0
        self.hits = 1
        
    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        
    def update(self, bbox):
        self.kf.update(bbox_to_z(bbox))
        self.time_since_update = 0
        self.hits += 1
        
    def get_bbox(self):
        return self.kf.get_state_bbox()
        
    def get_velocity(self):
        return self.kf.get_velocity()

class SimpleTracker:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        for trk in self.trackers:
            trk.predict()

        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]
        tracker_bboxes = [trk.get_bbox() for trk in self.trackers]
        
        matched_indices = []
        unmatched_detections = []
        
        if len(tracker_bboxes) == 0:
            unmatched_detections = list(range(len(detections)))
        elif len(detections) == 0:
            unmatched_detections = []
        else:
            cost_matrix = np.zeros((len(tracker_bboxes), len(detections)))
            for t, trk_bbox in enumerate(tracker_bboxes):
                for d, det_bbox in enumerate(detections):
                    iou = calculate_iou(trk_bbox, det_bbox)
                    cost_matrix[t, d] = -iou

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            unmatched_trackers = set(range(len(tracker_bboxes)))
            unmatched_dets_set = set(range(len(detections)))
            
            for r, c in zip(row_ind, col_ind):
                if -cost_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    unmatched_trackers.remove(r)
                    unmatched_dets_set.remove(c)
            
            unmatched_detections = list(unmatched_dets_set)

        for trk_idx, det_idx in matched_indices:
            self.trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_detections:
            trk = TrackedObject(detections[det_idx])
            self.trackers.append(trk)

        result_objs = []
        for trk in self.trackers:
            # 放宽显示条件：允许丢失 1 帧内的目标继续显示，防止闪烁
            if trk.time_since_update <= 1 and trk.hits >= self.min_hits:
                result_objs.append({
                    "id": trk.id, 
                    "bbox": trk.get_bbox(),
                    "vx": trk.get_velocity()[0],
                    "vy": trk.get_velocity()[1]
                })
        
        return result_objs

# ============================================================
# 测距工具与土豆形物理制动模型
# ============================================================

def bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)

def compute_homography_matrix():
    return cv2.getPerspectiveTransform(CALIB_PIXEL_POINTS, CALIB_REAL_POINTS)

def pixel_to_ground(point, H):
    pt = np.float32([[point[0], point[1]]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, H)
    return (transformed[0][0][0], transformed[0][0][1])

def calculate_dynamic_braking_distance(vx_px, vy_px, p_pixel, H):
    """
    根据车辆在像素空间的速度矢量，映射到真实世界计算 ISO 制动距离，
    并返回该方向延伸制动距离在像素画面上的投影结束点坐标
    """
    # 如果像素速度过小，认为是静止噪音，死区增大到 2.5 像素抑制抖动
    if math.hypot(vx_px, vy_px) < 2.5:
        return PHYSICS["MIN_SAFE_RADIUS"], p_pixel, 0.0

    # 我们通过模拟1帧后的点来算出物理速度的方向和大小
    p_next_pixel = (p_pixel[0] + vx_px, p_pixel[1] + vy_px)
    
    # 变换到真实地面坐标系
    p_ground = pixel_to_ground(p_pixel, H)
    p_next_ground = pixel_to_ground(p_next_pixel, H)
    
    # 真实世界的单帧运动向量和单帧距离（米）
    vector_x = p_next_ground[0] - p_ground[0]
    vector_y = p_next_ground[1] - p_ground[1]
    dist_per_frame = math.hypot(vector_x, vector_y)
    
    # 将速度统一换算至 m/s (乘以 FPS)
    v_real = dist_per_frame * PHYSICS["FPS"]
    
    # 算太快了或者是测距漂移，可加个上限(例如最大10m/s，约36km/h)
    v_real = min(v_real, 10.0) 

    # 如果物理速度低于 0.2 m/s 基本视作静止
    if v_real < 0.2:
        return PHYSICS["MIN_SAFE_RADIUS"], p_pixel, 0.0

    # ---------------- 核心制动距离公式 ----------------
    # 反应距离 = 速度 * 反应时间
    react_dist = v_real * PHYSICS["T_REACTION"]
    # 物理制动距离 = 速度平方 / (2 * 摩擦力系数 * g)
    brake_dist = (v_real ** 2) / (2 * PHYSICS["MU"] * PHYSICS["G"])
    # 动态总制动距离
    D_dynamic = PHYSICS["MIN_SAFE_RADIUS"] + react_dist + brake_dist
    # --------------------------------------------------

    # 计算出延伸点在真实坐标系的坐标
    # 因为矢量 (vector_x, vector_y) 是方向向量，距离是 dist_per_frame
    direction_x = vector_x / dist_per_frame
    direction_y = vector_y / dist_per_frame

    # 顺着车辆的运动方向向前延伸 D_dynamic 长度
    real_extend_x = p_ground[0] + direction_x * D_dynamic
    real_extend_y = p_ground[1] + direction_y * D_dynamic
    
    # 使用反变换将真实世界的制动终点重新映射回画面的像素坐标系
    H_inv = np.linalg.inv(H)
    pt_real = np.float32([[real_extend_x, real_extend_y]]).reshape(-1, 1, 2)
    transformed_back = cv2.perspectiveTransform(pt_real, H_inv)
    p_end_pixel = (int(transformed_back[0][0][0]), int(transformed_back[0][0][1]))

    return D_dynamic, p_end_pixel, v_real


def mutual_exclusion_model(person_p_real, vehicle_p_real, dynamic_danger_dist):
    """
    判断人的脚点是否侵入了车辆的动态制动危险区
    （严格来说应该判断多边形包含关系，这里简化为判断距离是否小于此时延伸出的最大防撞距离）
    * 对于土豆形包络线，我们假设如果距离车体小于危险距离即为入侵
    """
    d_real = math.hypot(person_p_real[0] - vehicle_p_real[0], person_p_real[1] - vehicle_p_real[1])

    if d_real <= dynamic_danger_dist:
        state = SystemState.DANGER
    elif d_real <= dynamic_danger_dist + PHYSICS["WARNING_MARGIN"]:
        state = SystemState.WARNING
    else:
        state = SystemState.SAFE

    return d_real, state

# ============================================================
# 可视化工具: 画土豆形预警包络线
# ============================================================
def draw_potato_envelope(frame, p_start, p_end, danger_dist, state, H):
    """
    在地面上绘制一个连接 start 和 end 的填充类椭圆形作为动态包络线。
    此处使用真实环境坐标系的圆点逆透视映射到像素系，形成带有透视畸变的真彩色贴地效果。
    """
    color = STATE_COLOR[state]
    overlay = frame.copy()
    
    # 获取车辆真实地面坐标
    pt1_g = pixel_to_ground(p_start, H)
    
    H_inv = np.linalg.inv(H)
    
    # 辅助函数：根据真实地面的圆心和半径生成透视多边形点集
    def get_perspective_circle_pts(center_g, radius_m, num_pts=36):
        circle_pts_g = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts
            # 在真实的俯视米制坐标上画圆
            x = center_g[0] + radius_m * math.cos(angle)
            y = center_g[1] + radius_m * math.sin(angle)
            circle_pts_g.append([x, y])
            
        pts_g_arr = np.float32(circle_pts_g).reshape(-1, 1, 2)
        # 用逆透视矩阵转换回画面里的畸变点
        pts_p_arr = cv2.perspectiveTransform(pts_g_arr, H_inv)
        # 取整返回
        return np.int32(pts_p_arr)

    # 1. 如果几乎静止，要求不再绘制静止小圆，留白即可
    if p_start == p_end or math.hypot(p_end[0]-p_start[0], p_end[1]-p_start[1]) < 10:
        pass
        
    # 2. 如果在运动，则画出起始圆到终点圆之间的多边形包络
    else:
        pt2_g = pixel_to_ground(p_end, H)
        
        # 两端的透视圆点集
        start_circle_pts = get_perspective_circle_pts(pt1_g, PHYSICS["MIN_SAFE_RADIUS"])
        # 远端圆可以适当放大或保持跟 danger_dist 匹配的扩展
        end_circle_pts = get_perspective_circle_pts(pt2_g, PHYSICS["MIN_SAFE_RADIUS"] * 1.2)
        
        # 为了形成包含线，我们可以把两个圆中心连线的垂线方向找到外包多边形
        # 更简单的作法：直接计算像素极值凸包，或者合并两个圆的所有点求凸包
        all_pts = np.vstack((start_circle_pts, end_circle_pts))
        hull = cv2.convexHull(all_pts)
        
        cv2.fillPoly(overlay, [hull], color)
        
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # 画外框与连接线
        cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=2)
        cv2.line(frame, (int(p_start[0]), int(p_start[1])), (int(p_end[0]), int(p_end[1])), (255, 255, 255), 2)
    
    # 标示立脚中心
    cv2.circle(frame, (int(p_start[0]), int(p_start[1])), 6, (0, 0, 255), -1)

# ============================================================
# 分析与包含关系过滤工具
# ============================================================

def is_inside(inner_box, outer_box):
    """判断 inner_box 的中心是否在 outer_box 内部"""
    ix_c = (inner_box[0] + inner_box[2]) / 2
    iy_c = (inner_box[1] + inner_box[3]) / 2
    
    return (outer_box[0] <= ix_c <= outer_box[2] and 
            outer_box[1] <= iy_c <= outer_box[3])

def filter_nested_boxes(persons, vehicles):
    """
    1. 过滤掉其实是驾驶员的'人' (人的框被包在车框里)
    2. 过滤掉重叠极高的大车套小车框 (如果IoU或者彼此包含)
    """
    filtered_persons = []
    # 过滤人：如果在任意一辆车内部，则很可能是驾驶员或车体部件
    for p in persons:
        is_driver = False
        for v in vehicles:
            if is_inside(p, v):
                is_driver = True
                break
        if not is_driver:
            filtered_persons.append(p)
            
    filtered_vehicles = []
    # 过滤重叠车辆框：如果一个车框被另一个更大的车框完全包含或极大比例相交，保留大框
    # 这里通过简单包含关系判断
    vehicles.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True) # 按面积降序排序
    for i, v1 in enumerate(vehicles):
        keep = True
        for j in range(i):
            v2 = vehicles[j] # v2 是更大的框
            if is_inside(v1, v2) or calculate_iou(v1, v2) > 0.6:
                keep = False
                break
        if keep:
            filtered_vehicles.append(v1)
            
    return filtered_persons, filtered_vehicles

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    model = YOLO("best.pt")
    
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

            raw_persons = []
            raw_vehicles = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                bbox = box.xyxy[0].tolist()

                # ==== 目标尺寸过滤逻辑 (只提取画面中的人和叉车) ====
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                
                # 去除太小的误报噪点
                if h < 10 or w < 10: 
                    continue

                aspect_ratio = h / w
                
                # 如果长宽比超过 1.5，形态非常修长，判定为“人”(之前 1.1 会误杀正面开来的叉车)
                if aspect_ratio >= 1.5:
                    raw_persons.append(bbox)
                # 否则形态扁平或方正，判定为“物流车/叉车”
                else:
                    raw_vehicles.append(bbox)
                    
            # --- 剔除重复/包含的目标 ---
            raw_persons, raw_vehicles = filter_nested_boxes(raw_persons, raw_vehicles)

            # --- 卡尔曼更新 ---
            smoothed_persons = person_tracker.update(raw_persons)
            smoothed_vehicles = vehicle_tracker.update(raw_vehicles)

            # --- 先清算并画出所有车辆的动态预警区土豆包络线 ---
            vehicle_danger_info = {}
            for v in smoothed_vehicles:
                v_p_pixel = bbox_bottom_center(v["bbox"])
                v_p_real = pixel_to_ground(v_p_pixel, H)

                # 使用卡尔曼的 (vx, vy) 计算物理制动前伸距离和像素结束点
                D_dynamic, extend_p_pixel, v_real = calculate_dynamic_braking_distance(
                    v["vx"], v["vy"], v_p_pixel, H
                )
                
                # 记录该车辆此时此刻需要的安全距离供互斥使用
                vehicle_danger_info[v["id"]] = {
                    "p_real": v_p_real,
                    "p_pixel": v_p_pixel,
                    "extend_p_pixel": extend_p_pixel,
                    "D_dynamic": D_dynamic,
                    "v_real": v_real,
                    "bbox": v["bbox"]
                }
                
                # 绘制车辆绿框 (先全画绿，如果有冲突后面再改红色)
                vx1, vy1, vx2, vy2 = map(int, v["bbox"])
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), STATE_COLOR[SystemState.SAFE], 2)
                # 车辆上方显示计算出的真实物理速度
                cv2.putText(frame, f"V: {v_real:.1f}m/s", (vx1, vy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # --- 再计算人车是否互斥入侵 ---
            for p in smoothed_persons:
                p_p_pixel = bbox_bottom_center(p["bbox"])
                p_p_real = pixel_to_ground(p_p_pixel, H)
                
                person_state = SystemState.SAFE
                
                # 只要和任何一辆车有冲突，人员就变色报警
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
                            
                # 绘制人员红色或绿色框 (人本身没有半透明小圆，只有在发生报警时框体变颜色)
                px1, py1, px2, py2 = map(int, p["bbox"])
                cv2.rectangle(frame, (px1, py1), (px2, py2), STATE_COLOR[person_state], 2)
                cv2.circle(frame, (int(p_p_pixel[0]), int(p_p_pixel[1])), 4, (0, 0, 255), -1)

            # --- 最后单独渲染所有的车及其冲突后的包络线 (先画底盘，防遮挡) ---
            for v_id, v_info in vehicle_danger_info.items():
                # 查询这辆车当前遭遇到的最高危人群状态
                max_v_state = SystemState.SAFE
                for p in smoothed_persons:
                    p_p_real = pixel_to_ground(bbox_bottom_center(p["bbox"]), H)
                    _, s = mutual_exclusion_model(p_p_real, v_info["p_real"], v_info["D_dynamic"])
                    if s > max_v_state:
                         max_v_state = s
                         
                # 绘制带透视的土豆形
                draw_potato_envelope(
                    frame, v_info["p_pixel"], v_info["extend_p_pixel"], 
                    v_info["D_dynamic"], max_v_state, H
                )

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
