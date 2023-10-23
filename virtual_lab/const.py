import enum

class LogLevel(enum.IntEnum):
    DIAGNOSTIC = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5 
class ColorCodingStrategies(enum.IntEnum):
    CYCLE = 1
    MONOCROME = 2

simulation_message_handling = {
    MsgTypes.MODEL_ADD_VAR: "_add_colors"
}