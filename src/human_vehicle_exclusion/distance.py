import math
import cv2
import numpy as np
from .config import PIXELS_PER_METER, DISTANCE_MODE, CALIB_PIXEL_POINTS, CALIB_REAL_POINTS

def bbox_bottom_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2)/2, y2)

def pixel_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def compute_homography_matrix():
    H = cv2.getPerspectiveTransform(CALIB_PIXEL_POINTS, CALIB_REAL_POINTS)
    return H

def pixel_to_ground(point, H):
    pt = np.float32([[point[0], point[1]]]).reshape(-1,1,2)
    transformed = cv2.perspectiveTransform(pt, H)
    return (transformed[0][0][0], transformed[0][0][1])

def estimate_distance_scale(person_bbox, vehicle_bbox):
    p1 = bbox_bottom_center(person_bbox)
    p2 = bbox_bottom_center(vehicle_bbox)
    d_pixel = pixel_distance(p1, p2)
    d_real = d_pixel / PIXELS_PER_METER
    return d_real, p1, p2

def estimate_distance_homography(person_bbox, vehicle_bbox, H):
    p1_pixel = bbox_bottom_center(person_bbox)
    p2_pixel = bbox_bottom_center(vehicle_bbox)
    p1_ground = pixel_to_ground(p1_pixel, H)
    p2_ground = pixel_to_ground(p2_pixel, H)
    d_real = math.hypot(p1_ground[0] - p2_ground[0], p1_ground[1] - p2_ground[1])
    return d_real, p1_pixel, p2_pixel

def compute_real_distance(person_bbox, vehicle_bbox, H=None):
    if DISTANCE_MODE == "homography" and H is not None:
        return estimate_distance_homography(person_bbox, vehicle_bbox, H)
    else:
        return estimate_distance_scale(person_bbox, vehicle_bbox)
