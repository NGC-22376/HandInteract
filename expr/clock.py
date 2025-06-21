import platform
import os
import tkinter as tk
from threading import Thread
from time import sleep


def beep():
    """根据不同操作系统发出蜂鸣声"""
    system = platform.system()
    if system == "Windows":
        import winsound
        # 发出标准蜂鸣声
        winsound.MessageBeep()
    elif system == "Linux" or system == "Darwin":  # macOS
        # 使用 shell 命令发出蜂鸣声
        os.system('echo -e "\a"')
    else:
        print("不支持当前操作系统的蜂鸣功能")


def monitor_clicks():
    """使用tkinter监听鼠标点击"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    def on_click(event):
        if event.num == 1:  # 左键点击
            beep()
            print(f"左键点击位置: ({event.x_root}, {event.y_root})")

    # 创建一个全屏透明窗口来捕获鼠标事件
    overlay = tk.Toplevel(root)
    overlay.attributes('-fullscreen', True)
    overlay.attributes('-alpha', 0.0)  # 完全透明
    overlay.bind('<Button-1>', on_click)  # 绑定左键点击事件

    # 按ESC键退出程序
    overlay.bind('<Escape>', lambda e: overlay.quit())

    # 设置窗口置顶
    overlay.attributes('-topmost', True)

    # 启动消息循环
    root.mainloop()


def main():
    """主函数：启动鼠标监听"""
    print("程序已启动，正在监听鼠标左键点击...")
    print("按ESC键终止程序")

    # 在单独的线程中运行tkinter事件循环
    monitor_thread = Thread(target=monitor_clicks)
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # 保持主线程运行
        while True:
            sleep(1)
    except KeyboardInterrupt:
        print("\n程序已终止")


if __name__ == "__main__":
    main()