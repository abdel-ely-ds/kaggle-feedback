import os
import time


def shutdown(threshold=3.5):
    while time.gmtime().tm_hour < threshold:
        time.sleep(600)  # wait 5 minutes
    os.system("shutdown /s /t 90")


if __name__ == '__main__':
    shutdown()
