
"""
目标追踪模块
- 卡尔曼滤波：BBoxKalmanFilter(框卡尔曼滤波)
- 追踪对象：TrackedObject(单个追踪目标)
- 多目标追踪：SimpleTracker(基于IOU的多目标追踪器)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        self.Q[0:self.ndim, 0:self.ndim] *= 0.005 
        self.Q[self.ndim:, self.ndim:] *= 0.00001
        
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
            if trk.time_since_update <= 1 and trk.hits >= self.min_hits:
                result_objs.append({
                    "id": trk.id, 
                    "bbox": trk.get_bbox(),
                    "vx": trk.get_velocity()[0],
                    "vy": trk.get_velocity()[1]
                })
        
        return result_objs