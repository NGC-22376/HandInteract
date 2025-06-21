import json
import os.path
import time
import pyautogui
import tkinter as tk
import winsound
import sys

class AutoClicker:
    def __init__(self):
        self.begin = False
        self.window = None

        # 初始化坐标
        self.path = os.path.join(os.path.expanduser("~"), "Documents", "coordinate.json")
        self.load_coordinates()

        # 创建GUI
        self.create_gui()

    def load_coordinates(self):
        with open(self.path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        # 提取坐标
        self.x1, self.y1 = self.data["btn1"]["x"], self.data["btn1"]["y"]
        self.x2, self.y2 = self.data["btn2"]["x"], self.data["btn2"]["y"]
        self.x3, self.y3 = self.data["btn3"]["x"], self.data["btn3"]["y"]
        print(self.x1, self.y1, self.x2, self.y2, self.x3, self.y3)

    def create_gui(self):
        self.window = tk.Tk()
        self.window.title("自动点击器")
        self.window.geometry("80x40")  # 80*20的窗口
        self.window.resizable(False, False)
        self.window.attributes('-topmost', True)

        # 创建按钮
        self.start_button = tk.Button(self.window, text="启动", command=self.toggle_clicking)
        self.start_button.pack(fill=tk.BOTH, expand=True)

        # 窗口关闭时退出程序
        self.window.protocol("WM_DELETE_WINDOW", self.exit_program)

    def toggle_clicking(self):
        if not self.begin:
            winsound.Beep(frequency=1500, duration=200)  # 开始提示音
            pyautogui.click(self.x1, self.y1)
            time.sleep(0.1)
            winsound.Beep(frequency=1500, duration=200)
            pyautogui.click(self.x2, self.y2)
        else:
            winsound.Beep(frequency=1000, duration=200)  # 结束提示音
            pyautogui.click(self.x1, self.y1)
            time.sleep(0.1)
            winsound.Beep(frequency=1000, duration=200)
            pyautogui.click(self.x3, self.y3)

        self.begin = not self.begin

    def exit_program(self):
        # 关闭窗口
        if self.window:
            self.window.destroy()

        # 退出程序
        sys.exit()

    def run(self):
        # 启动GUI主循环
        self.window.mainloop()

if __name__ == '__main__':
    # 禁用PyAutoGUI的安全特性，防止鼠标移动到角落时中断
    pyautogui.FAILSAFE = False

    # 创建并运行自动点击器
    auto_clicker = AutoClicker()
    auto_clicker.run()