from .distance import compute_real_distance
from .system_state import SystemState
from .config import ALARM_DISTANCE_M, WARNING_DISTANCE_M

def mutual_exclusion_model(person_bbox, vehicle_bbox, H=None):
    d_real, p1, p2 = compute_real_distance(person_bbox, vehicle_bbox, H)
    if d_real <= ALARM_DISTANCE_M:
        state = SystemState.DANGER
    elif d_real <= WARNING_DISTANCE_M:
        state = SystemState.WARNING
    else:
        state = SystemState.SAFE
    return d_real, state, p1, p2
