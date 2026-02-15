from src.core import main

if __name__ == "__main__":
    """
    简单演示入口：
    运行本脚本即可启动摄像头，加载 YOLO 模型，
    并执行人车互斥检测 + 报警 + 可视化 + MQTT 发布
    """
    main()
