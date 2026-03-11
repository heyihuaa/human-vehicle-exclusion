"""
核心计算模块
- 坐标转换：bbox_bottom_center(框底中心点)、compute_homography_matrix(单应性矩阵)、pixel_to_ground(像素转真实坐标)
- 距离计算：calculate_dynamic_braking_distance(动态制动距离)
- 互斥判断：mutual_exclusion_model(人车互斥模型)
- 可视化：draw_potato_envelope(绘制预警区域)
"""
import math
import numpy as np
import cv2
from config import PHYSICS, CALIB_PIXEL_POINTS, CALIB_REAL_POINTS, SystemState

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
    if math.hypot(vx_px, vy_px) < 2.5:
        return PHYSICS["MIN_SAFE_RADIUS"], p_pixel, 0.0

    p_next_pixel = (p_pixel[0] + vx_px, p_pixel[1] + vy_px)
    p_ground = pixel_to_ground(p_pixel, H)
    p_next_ground = pixel_to_ground(p_next_pixel, H)

    vector_x = p_next_ground[0] - p_ground[0]
    vector_y = p_next_ground[1] - p_ground[1]
    dist_per_frame = math.hypot(vector_x, vector_y)

    v_real = dist_per_frame * PHYSICS["FPS"]
    v_real = min(v_real, 10.0)

    if v_real < 0.2:
        return PHYSICS["MIN_SAFE_RADIUS"], p_pixel, 0.0

    react_dist = v_real * PHYSICS["T_REACTION"]
    brake_dist = (v_real ** 2) / (2 * PHYSICS["MU"] * PHYSICS["G"])
    D_dynamic = PHYSICS["MIN_SAFE_RADIUS"] + react_dist + brake_dist

    direction_x = vector_x / dist_per_frame
    direction_y = vector_y / dist_per_frame

    real_extend_x = p_ground[0] + direction_x * D_dynamic
    real_extend_y = p_ground[1] + direction_y * D_dynamic

    H_inv = np.linalg.inv(H)
    pt_real = np.float32([[real_extend_x, real_extend_y]]).reshape(-1, 1, 2)
    transformed_back = cv2.perspectiveTransform(pt_real, H_inv)
    p_end_pixel = (int(transformed_back[0][0][0]), int(transformed_back[0][0][1]))

    return D_dynamic, p_end_pixel, v_real

def mutual_exclusion_model(person_p_real, vehicle_p_real, dynamic_danger_dist):
    d_real = math.hypot(person_p_real[0] - vehicle_p_real[0], person_p_real[1] - vehicle_p_real[1])

    if d_real <= dynamic_danger_dist:
        state = SystemState.DANGER
    elif d_real <= dynamic_danger_dist + PHYSICS["WARNING_MARGIN"]:
        state = SystemState.WARNING
    else:
        state = SystemState.SAFE

    return d_real, state

def draw_potato_envelope(frame, p_start, p_end, danger_dist, state, H):
    color = STATE_COLOR[state]
    overlay = frame.copy()
    pt1_g = pixel_to_ground(p_start, H)
    H_inv = np.linalg.inv(H)

    def get_perspective_circle_pts(center_g, radius_m, num_pts=36):
        circle_pts_g = []
        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts
            x = center_g[0] + radius_m * math.cos(angle)
            y = center_g[1] + radius_m * math.sin(angle)
            circle_pts_g.append([x, y])
        pts_g_arr = np.float32(circle_pts_g).reshape(-1, 1, 2)
        pts_p_arr = cv2.perspectiveTransform(pts_g_arr, H_inv)
        return np.int32(pts_p_arr)

    if p_start == p_end or math.hypot(p_end[0]-p_start[0], p_end[1]-p_start[1]) < 10:
        pass
    else:
        pt2_g = pixel_to_ground(p_end, H)
        start_circle_pts = get_perspective_circle_pts(pt1_g, PHYSICS["MIN_SAFE_RADIUS"])
        end_circle_pts = get_perspective_circle_pts(pt2_g, PHYSICS["MIN_SAFE_RADIUS"] * 1.2)
        all_pts = np.vstack((start_circle_pts, end_circle_pts))
        hull = cv2.convexHull(all_pts)
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=2)
        cv2.line(frame, (int(p_start[0]), int(p_start[1])), (int(p_end[0]), int(p_end[1])), (255, 255, 255), 2)

    cv2.circle(frame, (int(p_start[0]), int(p_start[1])), 6, (0, 0, 255), -1)

# 为了在 draw_potato_envelope 中使用 STATE_COLOR，需要从 config 导入
from config import STATE_COLOR