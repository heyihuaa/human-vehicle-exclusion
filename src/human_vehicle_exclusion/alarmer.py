import time
import yagmail
from datetime import datetime
from .config import EMAIL_SETTING, ALARM_SETTING

LAST_ALARM = {}

def send_alarm_email(camera_id: str, detail: str) -> bool:
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
            print(f"[邮箱重试] 第{i+1}次失败：{e}")
            time.sleep(2)
    return False

def trigger_vehicle_person_alarm(camera_id: str, detail: str):
    now = time.time()
    if camera_id in LAST_ALARM:
        if now - LAST_ALARM[camera_id] < ALARM_SETTING["cool_down"]:
            print(f"[冷却] {camera_id} 冷却中，跳过告警")
            return
    LAST_ALARM[camera_id] = now
    email_ok = send_alarm_email(camera_id, detail)
    from .logger import write_alarm_log
    write_alarm_log(camera_id, detail, email_ok)
