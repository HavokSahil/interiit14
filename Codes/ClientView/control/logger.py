from control.controller import Controller
from enum import Enum

class LogLevel(Enum):
    EXCESSIVE = 0
    MSGDUMP = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5

    @staticmethod
    def int_to_level(level: int) -> "LogLevel":
        if level == 0:
            return LogLevel.EXCESSIVE
        elif level == 1:
            return LogLevel.MSGDUMP
        elif level == 2:
            return LogLevel.DEBUG
        elif level == 3:
            return LogLevel.INFO
        elif level == 4:
            return LogLevel.WARNING
        else:
            # NOTE: i know this can go wrong, still
            # i am having faith that this will not
            return LogLevel.ERROR


class Logger:
    @staticmethod
    def set_log_level(controller: Controller, level: LogLevel):
        return controller.send_command(f"LOG_LEVEL {level.name}")
    
    