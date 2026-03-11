"""
通用工具模块
- 几何计算：area(框面积)、calculate_iou(交并比)、is_fully_contained(框包含判断)
- 框转换：bbox_to_z(框转卡尔曼向量)、z_to_bbox(卡尔曼向量转框)
- 过滤逻辑：filter_person_in_forktruck(过滤叉车内部人员框)
"""
import json
from datetime import datetime
from config import ALARM_SETTING

LAST_ALARM = {}

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