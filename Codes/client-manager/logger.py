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
    
    # State variable for current allowed log level
    _current_log_level: LogLevel = LogLevel.INFO

    @classmethod
    def set_current_log_level(cls, level: LogLevel):
        cls._current_log_level = level

    @classmethod
    def get_current_log_level(cls) -> LogLevel:
        return cls._current_log_level

    @staticmethod
    def log(level: LogLevel, msg: str) -> None:
        if level.value >= Logger._current_log_level.value:
            print(f"[{level.name}] {msg}")

    @staticmethod
    def log_err(msg: str) -> None:
        Logger.log(LogLevel.ERROR, msg)
    
    @staticmethod
    def log_debug(msg: str) -> None:
        Logger.log(LogLevel.DEBUG, msg)
    
    @staticmethod
    def log_info(msg: str) -> None:
        Logger.log(LogLevel.INFO, msg)

