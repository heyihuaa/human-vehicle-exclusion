
# Human-Vehicle Mutual Exclusion System  
# 人车互斥实时监测与告警系统

基于 **YOLOv8** 的人车互斥实时监测系统，用于检测人员与车辆在同一画面中的危险接近行为，并在达到风险阈值时触发 **邮件告警** 与 **本地日志记录**。

---

## 功能特点

- 实时检测摄像头中的人员和车辆
- 根据像素距离和透视权重计算安全距离
- 风险等级分为 SAFE / WARNING / DANGER
- 危险状态自动触发：
  - 邮件告警
  - 本地日志记录
- 冷却时间机制避免重复告警
- OpenCV 实时可视化显示

---

## 环境要求

- Python 3.11+
- 依赖库：
  - `opencv-python`
  - `ultralytics`
  - `yagmail`

安装依赖命令：

```bash
pip install opencv-python ultralytics yagmail
````
---

## 项目文件结构

```text
├── final.py                     # 主程序
├── README.md                    # 项目说明文档
├── .gitignore                   # Git 忽略规则
├── yolov8n.pt                   # YOLOv8 模型文件（不提交）
└── vehicle_person_alarm.log     # 告警日志文件（运行生成）
```

### 文件说明表

| 文件名                        | 说明                   |
| -------------------------- | -------------------- |
| `final.py`                 | 系统主程序，负责检测、判断、告警与可视化 |
| `.gitignore`               | 忽略模型文件、日志文件及缓存       |
| `yolov8n.pt`               | YOLOv8 预训练模型（需自行下载）  |
| `vehicle_person_alarm.log` | 告警日志文件，JSON 行格式      |

---

## 使用方法

### 1. 邮箱告警配置（必须）

#### Windows

```bash
setx ALARM_EMAIL_SENDER "发送告警的邮箱"
setx ALARM_EMAIL_AUTH "邮箱SMTP授权码"
setx ALARM_EMAIL_RECEIVERS "接收邮箱1,接收邮箱2"
```

#### Linux / macOS

```bash
export ALARM_EMAIL_SENDER="发送告警的邮箱"
export ALARM_EMAIL_AUTH="邮箱SMTP授权码"
export ALARM_EMAIL_RECEIVERS="接收邮箱1,接收邮箱2"
```

> ⚠️ QQ 邮箱需开启 SMTP 并使用授权码，而非登录密码。

### 2. 启动程序

```bash
python final.py
```

* 默认使用本地摄像头（`VideoCapture(0)`）
* 程序启动后弹出实时检测窗口
* 按 `ctrl+c` 键退出程序

---

## 报警逻辑

1. 读取摄像头视频流
2. 使用 YOLOv8 检测目标
3. 分类提取人员与车辆
4. 计算 bbox 中心点距离
5. 根据 y 轴透视权重调整距离
6. 计算安全半径并判断风险等级
7. 若为 DANGER：

   * 发送邮件告警
   * 写入日志
8. 已触发告警的摄像头进入冷却时间，避免重复报警

---

## 人车互斥判定模型

* 使用 bbox 中心点作为目标位置
* 计算像素欧氏距离
* 使用 y 轴比例作为透视加权：

```text
d_real = d_pixel × perspective_weight(y)
```

* 安全半径 = 基础半径 + 速度因子
* 风险等级：

  * SAFE：安全
  * WARNING：预警
  * DANGER：危险，触发告警

---

## 可视化说明

| 状态      | 显示效果           |
| ------- | -------------- |
| SAFE    | 绿色边框           |
| WARNING | 黄色边框           |
| DANGER  | 红色边框           |
| 文本      | 显示在车辆边框上方，实时更新 |

---

## 日志说明

* 日志格式：JSON（每行一条）
* 字段说明：

  * `camera_id`：摄像头编号
  * `alarm_type`：告警类型
  * `alarm_time`：告警时间
  * `detail`：告警详情
  * `email_send_success`：邮件发送是否成功

示例：

```json
{
  "camera_id": "CAM_01",
  "alarm_type": "人车互斥",
  "alarm_time": "2026-02-07 15:45:12",
  "detail": "人员0 与车辆1 距离 52.3，低于安全阈值 80.0",
  "email_send_success": true
}
```

