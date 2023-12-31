import sys
from logging import Logger

class FileLogger(Logger):
    def __init__(self, filename):
        super().__init__(self)
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def info(self, message):
        self.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False
    
# sys.stdout = FileLogger(".output.log")