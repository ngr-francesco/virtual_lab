import enum

class LogLevel(enum.IntEnum):
    DIAGNOSTIC = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5 


class VL_Logger:

    # Static variable to hold the logging level.
    level = LogLevel.CRITICAL

    def __init__(self, name = None):
        """
        Initialize the logger in the DEBUG level.
        """
        self.name = name

    
    def set_level(self, level):
        """
        Set the logging level.
        """
        self.level = level

    def log(self, level, message):
        """
        Log a message at the specified level.
        """
        if self.level <= level:
            print(self.format_message(level, message))

    def format_message(self, level, message):
        """
        Format the message.
        """
        return f"{level.name} {self.name}: {message}"

    def diagnostic(self, message):
        """
        Log a message at the DIAGNOSTIC level.
        """
        self.log(LogLevel.DIAGNOSTIC, message)
    
    def debug(self, message):
        """
        Log a message at the DEBUG level.
        """
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message):
        """
        Log a message at the INFO level.
        """
        self.log(LogLevel.INFO, message)

    def warning(self, message):
        """
        Log a message at the WARNING level.
        """
        self.log(LogLevel.WARNING, message)
    
    def error(self, message):
        """
        Log a message at the ERROR level.
        """
        self.log(LogLevel.ERROR, message)
    
    def critical(self, message):
        """
        Log a message at the CRITICAL level.
        """
        self.log(LogLevel.CRITICAL, message)

    @staticmethod
    def set_log_level(level):
        """
        Finds any number of globally defined VL_Logger instances and sets their level.
        """
        if level in LogLevel:
            VL_Logger.level = level
            print(f"Set log level to {level.name}")
        else:
            raise ValueError(f"Invalid log level: {level}")

def get_logger(name):
    """
    Get a logger with the specified name.
    """
    logger =  VL_Logger(name)
    return logger

