import pyautogui
import os
import time


def keep_alive():
    pyautogui.FAILSAFE = False
    x, y = 1620, 120
    while True:
        pyautogui.click(x=x, y=y)
        time.sleep(60)
        if time.gmtime().tm_hour > 6:
            os.system("shutdown /s /t 90")






if __name__ == '__main__':
    keep_alive()
