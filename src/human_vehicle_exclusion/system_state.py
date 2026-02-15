class SystemState:
    SAFE = 0
    WARNING = 1
    DANGER = 2

STATE_COLOR = {
    SystemState.SAFE: (0, 255, 0),
    SystemState.WARNING: (0, 255, 255),
    SystemState.DANGER: (0, 0, 255)
}

STATE_TEXT = {
    SystemState.SAFE: "SAFE",
    SystemState.WARNING: "WARNING",
    SystemState.DANGER: "DANGER"
}
