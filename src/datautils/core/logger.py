import sys
import time

def print(msg, level="INFO"):
    if msg != "\n":
        msg = "[%s] %s: %s\n" % (time.asctime(), level, msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
