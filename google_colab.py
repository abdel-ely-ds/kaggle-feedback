import pyautogui
import time


def keep_alive():
    pyautogui.FAILSAFE = False
    x, y = 2000 + 1600, 120
    while True:
        pyautogui.click(x=x, y=y)
        time.sleep(60)


if __name__ == '__main__':
    keep_alive()
