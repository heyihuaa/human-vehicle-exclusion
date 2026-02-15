# 人车互斥安全监测系统

## 项目简介
你提供的这个项目是一套基于YOLOv8的实时人车互斥安全监测系统，能够通过摄像头实时检测画面中的行人和车辆，计算二者之间的实际距离，并在距离低于安全阈值时触发邮件告警和MQTT消息推送，同时具备车辆运动状态检测功能，适用于工厂、园区等需要人车安全管控的场景。

## 功能特性
- 🎯 **实时目标检测**：基于YOLOv8检测画面中的行人和车辆（truck/car/bus）
- 📏 **距离计算**：支持两种测距模式（比例尺模式/单应性矩阵模式）
- 🚨 **多级告警**：危险距离（1.5m）触发告警，警告距离（3.0m）仅视觉提示
- 📧 **邮件告警**：距离超限自动发送告警邮件（支持重试机制）
- 📡 **MQTT推送**：告警信息实时推送至MQTT服务器
- 🚗 **运动检测**：识别车辆运动/静止状态
- 📜 **日志记录**：完整记录告警信息和邮件发送状态
- 🎨 **可视化展示**：实时显示检测框、距离、告警状态和运动状态

## 环境要求
### 基础环境
- Python 3.8+
- OpenCV 4.x
- PyTorch 1.9+
- ultralytics (YOLOv8)

### 依赖安装
```bash
pip install opencv-python numpy ultralytics yagmail paho-mqtt python-dotenv
```

## 配置说明
### 1. 环境变量配置
建议创建 `.env` 文件配置敏感信息：
```env
# 邮件配置
ALARM_EMAIL_SENDER=你的QQ邮箱@qq.com
ALARM_EMAIL_AUTH=你的QQ邮箱授权码
ALARM_EMAIL_RECEIVERS=接收邮箱1@xxx.com,接收邮箱2@xxx.com
```

### 2. 核心参数配置
在代码中可调整以下关键配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ALARM_DISTANCE_M` | 1.5 | 危险告警距离（米） |
| `WARNING_DISTANCE_M` | 3.0 | 警告距离（米） |
| `PIXELS_PER_METER` | 100.0 | 比例尺模式下像素/米比例 |
| `ALARM_SETTING["cool_down"]` | 10 | 告警冷却时间（秒） |
| `MOTION_SETTING["motion_threshold"]` | 500 | 运动检测阈值 |
| `MQTT_BROKER` | localhost | MQTT服务器地址 |
| `MQTT_PORT` | 1883 | MQTT端口 |
| `DEVICE_ID` | FORK-001 | 设备编号 |

### 3. 测距模式配置
- **比例尺模式（默认）**：`DISTANCE_MODE = "scale"`
  - 通过 `PIXELS_PER_METER` 设置像素与实际米数的转换比例
- **单应性矩阵模式**：`DISTANCE_MODE = "homography"`
  - 需要根据实际场景调整 `CALIB_PIXEL_POINTS` 和 `CALIB_REAL_POINTS` 校准点

## 运行说明
### 基本运行
```bash
python 你的脚本名.py
```

### 操作说明
- 程序启动后会自动调用默认摄像头（ID=0）
- 按下 `q` 键退出程序
- 实时窗口显示：
  - 行人检测框（绿色）
  - 车辆检测框（绿色=安全/黄色=警告/红色=危险）
  - 人车距离标注
  - 车辆运动状态
  - 系统模式和告警阈值

## 核心功能模块说明
### 1. 目标检测模块
使用YOLOv8模型检测画面中的行人（person）和车辆（truck/car/bus），提取目标的边界框（bbox）信息。

### 2. 运动检测模块
- 对视频帧进行灰度化和高斯模糊预处理
- 通过帧差法计算车辆区域的像素变化
- 判断车辆是运动中还是静止状态（支持延迟判定）

### 3. 距离计算模块
- `bbox_bottom_center()`：计算目标边界框底部中心点
- `compute_real_distance()`：根据选定模式计算实际距离
- `mutual_exclusion_model()`：根据距离判断安全状态（安全/警告/危险）

### 4. 告警模块
- `send_alarm_email()`：发送告警邮件（支持3次重试）
- `trigger_vehicle_person_alarm()`：告警触发（含冷却机制）
- `write_alarm_log()`：记录告警日志到文件

### 5. MQTT通信模块
- 连接MQTT服务器并启动异步循环
- 危险状态下推送JSON格式告警信息

## 日志格式
告警日志保存在 `vehicle_person_alarm.log` 文件中，每条日志为JSON格式：
```json
{
  "camera_id": "CAM_01",
  "alarm_type": "人车互斥",
  "alarm_time": "2026-02-15 10:00:00",
  "detail": "人员0 与车辆0 实际距离 1.20m，低于安全阈值 1.5m",
  "email_send_success": true
}
```

## MQTT消息格式
告警时推送的MQTT消息payload格式：
```json
{
  "device_id": "FORK-001",
  "alarm": 1,
  "driver_present": 1,
  "outer_intrusion": 1,
  "timestamp": "2026-02-15 10:00:00"
}
```

## 注意事项
1. **QQ邮箱配置**：需要开启SMTP服务并获取授权码（不是登录密码）
2. **摄像头权限**：确保程序有权限访问摄像头设备
3. **MQTT服务器**：运行前确保MQTT服务器已启动并可连接
4. **测距校准**：根据实际场景调整测距参数以保证精度
5. **模型文件**：首次运行会自动下载YOLOv8n.pt模型文件
6. **告警冷却**：避免短时间内重复发送告警

## 扩展建议
1. 支持多摄像头同时监测
2. 添加视频文件输入支持（而非仅摄像头）
3. 增加Web界面展示实时画面和告警记录
4. 支持更多车辆类型识别（如叉车、摩托车等）
5. 添加声音告警功能
6. 优化单应性矩阵校准流程，支持手动标定
7. 增加历史数据统计和可视化功能

## 故障排查
- **摄像头无法打开**：检查摄像头是否被占用，或修改 `cv2.VideoCapture(0)` 中的设备ID
- **邮件发送失败**：检查邮箱授权码、SMTP服务器配置和网络连接
- **MQTT连接失败**：检查MQTT服务器地址、端口和防火墙设置
- **测距不准确**：重新校准 `PIXELS_PER_METER` 或单应性矩阵校准点
- **检测效果差**：可更换更大的YOLO模型（如yolov8s.pt、yolov8m.pt）
