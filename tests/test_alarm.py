import time
from src.alarmer import trigger_vehicle_person_alarm
from src.utils import LAST_ALARM
from src.config import ALARM_SETTING

def test_alarm_trigger():
    """
    模拟一次告警触发，测试日志和冷却机制
    """
    camera_id = "TEST_CAM"
    detail = "测试人员与车辆距离过近"

    print("第一次触发告警（应该发送邮件并写日志）")
    trigger_vehicle_person_alarm(camera_id, detail)
    time.sleep(1)

    print("第二次触发告警（小于冷却时间，不应该重复发送）")
    trigger_vehicle_person_alarm(camera_id, detail)
    print(f"LAST_ALARM dict: {LAST_ALARM}")

if __name__ == "__main__":
    test_alarm_trigger()
