from datetime import datetime
import time
import json
import yagmail

EMAIL_SETTING = {
    "smtp_server": "smtp.qq.com",
    "sender": "3403489687@qq.com",
    "auth_code": "ibiqexfhuglidbdd",
    "receivers": ["yuz11_4781@qq.com"]
}

ALARM_SETTING = {
    "log_file": "vehicle_person_alarm.log"
}

def write_alarm_log(camera_id, detail, email_ok):
    log_info = {
        "camera_id": camera_id,
        "alarm_type": "人车互斥",
        "alarm_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detail": detail,
        "email_send_success": email_ok
    }
    with open(ALARM_SETTING["log_file"], "a", encoding="utf-8") as f:
        f.write(json.dumps(log_info, ensure_ascii=False) + "\n")
    print("[日志] 写入成功")

def send_alarm_email():
    yag = yagmail.SMTP(
        user=EMAIL_SETTING["sender"],
        password=EMAIL_SETTING["auth_code"],
        host=EMAIL_SETTING["smtp_server"],
        port=465,
        smtp_ssl=True
    )
    yag.send(
        to=EMAIL_SETTING["receivers"],
        subject="【测试邮件】人车互斥系统",
        contents="如果你收到这封邮件，说明报警模块完全正常。"
    )
    print("[邮箱] 邮件已发送")

if __name__ == "__main__":
    print("开始测试报警模块")
    try:
        send_alarm_email()
        write_alarm_log("TEST_CAM", "这是一次纯邮件测试", True)
    except Exception as e:
        print("发生错误：", e)
