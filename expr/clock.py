import platform

from pynput.mouse import Listener, Button

# 根据操作系统选择不同的声音输出方式
try:
    import winsound  # Windows系统
except ImportError:
    import os  # 非Windows系统备用方案

# 目标区域定义 (请根据实际需要修改坐标)
TARGET_X = 395  # 区域左上角X坐标
TARGET_Y = 100  # 区域左上角Y坐标
WIDTH = 60  # 区域宽度
HEIGHT = 30  # 区域高度


def beep():
    """蜂鸣提示音"""
    try:
        # Windows系统使用winsound
        winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
    except:
        # 其他系统尝试使用系统命令
        if platform.system() == 'Darwin':  # macOS
            os.system('say "beep"')
        elif platform.system() == 'Linux':  # Linux
            os.system('play -nq -t alsa synth 0.5 sine 1000 2>/dev/null')


def on_click(x, y, button, pressed):
    """鼠标点击事件处理"""
    if pressed and button == Button.left:
        print(f"检测到点击坐标: ({x}, {y})")  # 调试信息

        # 检查是否在目标区域内
        if (TARGET_X <= x <= TARGET_X + WIDTH and
                TARGET_Y <= y <= TARGET_Y + HEIGHT):
            print("命中目标区域！")
            beep()


def main():
    # 启动鼠标监听器
    with Listener(on_click=on_click) as listener:
        print(f"监听已启动，目标区域：({TARGET_X}, {TARGET_Y}) 到 "
              f"({TARGET_X + WIDTH}, {TARGET_Y + HEIGHT})")
        print("点击任意位置可查看坐标，点击目标区域将触发蜂鸣")
        print("按 Ctrl+C 退出程序...")
        listener.join()


if __name__ == "__main__":
    main()
