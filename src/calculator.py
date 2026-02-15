import math
import numpy as np
from .config import DISTANCE_MODE, PIXELS_PER_METER, ALARM_DISTANCE_M, WARNING_DISTANCE_M
from .config import CALIB_PIXEL_POINTS, CALIB_REAL_POINTS, SystemState

def bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2)/2, y2)

def pixel_distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def compute_real_distance(person_bbox, vehicle_bbox, H=None):
    if DISTANCE_MODE == "homography" and H is not None:
        pt1 = pixel_to_ground(bbox_bottom_center(person_bbox), H)
        pt2 = pixel_to_ground(bbox_bottom_center(vehicle_bbox), H)
        d_real = math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])
        return d_real, bbox_bottom_center(person_bbox), bbox_bottom_center(vehicle_bbox)
    else:
        d_pixel = pixel_distance(bbox_bottom_center(person_bbox), bbox_bottom_center(vehicle_bbox))
        d_real = d_pixel / PIXELS_PER_METER
        return d_real, bbox_bottom_center(person_bbox), bbox_bottom_center(vehicle_bbox)

def compute_homography_matrix():
    import cv2
    return cv2.getPerspectiveTransform(CALIB_PIXEL_POINTS, CALIB_REAL_POINTS)

def pixel_to_ground(point, H):
    import cv2
    pt = np.float32([[point[0], point[1]]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, H)
    return (transformed[0][0][0], transformed[0][0][1])

def mutual_exclusion_model(person_bbox, vehicle_bbox, H=None):
    d_real, pt_person, pt_vehicle = compute_real_distance(person_bbox, vehicle_bbox, H)
    if d_real <= ALARM_DISTANCE_M:
        state = SystemState.DANGER
    elif d_real <= WARNING_DISTANCE_M:
        state = SystemState.WARNING
    else:
        state = SystemState.SAFE
    return d_real, state, pt_person, pt_vehicle
