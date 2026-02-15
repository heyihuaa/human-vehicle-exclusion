import cv2
from .system_state import STATE_COLOR, STATE_TEXT

def draw_frame(frame, person, vehicle, state, d_real, pt_person, pt_vehicle, person_idx, vehicle_idx):
    # 人员 bbox
    px1, py1, px2, py2 = map(int, person["bbox"])
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0,255,0), 2)
    cv2.putText(frame, f"Person {person_idx}", (px1, py1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 车辆 bbox
    vx1, vy1, vx2, vy2 = map(int, vehicle["bbox"])
    color = STATE_COLOR[state]
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2)
    cv2.putText(frame, STATE_TEXT[state], (vx1, vy1-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"{d_real:.2f}m", (vx1, vy1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 连线
    pt_p = (int(pt_person[0]), int(pt_person[1]))
    pt_v = (int(pt_vehicle[0]), int(pt_vehicle[1]))
    cv2.line(frame, pt_p, pt_v, color, 2)
    mid_x = (pt_p[0] + pt_v[0])//2
    mid_y = (pt_p[1] + pt_v[1])//2
    cv2.putText(frame, f"{d_real:.2f}m", (mid_x+5, mid_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
