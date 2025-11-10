from control.controller import Controller
from control.logger import *

def test_logger_level_set():
    controller = Controller()
    controller.connect()
    assert(Logger.set_log_level(controller, LogLevel.EXCESSIVE))
    assert(Logger.set_log_level(controller, LogLevel.MSGDUMP))
    assert(Logger.set_log_level(controller, LogLevel.INFO))
    assert(Logger.set_log_level(controller, LogLevel.WARNING))
    assert(Logger.set_log_level(controller, LogLevel.ERROR))
    controller.disconnect()

def test_all_logger():
    test_logger_level_set()