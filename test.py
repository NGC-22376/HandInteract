import os
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

path1 = r"E:\UESTC\手互\Dataset\Dataset-2\李佳乐\H"  # 数据集路径
path2 = r"E:\UESTC\手互\output"  # 处理后的信号保存路径

# 确保输出目录存在
os.makedirs(path2, exist_ok=True)

# 获取所有 CSV 文件路径
data_files = []
for root, _, files in os.walk(path1):
    for file in files:
        if file.endswith(".csv"):  # 查找所有 CSV 文件
            data_files.append(os.path.join(root, file))

data_files.sort()  # 确保遍历顺序

current_index = 0  # 当前文件索引


# 读取 CSV
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, skiprows=4, names=["time", "voltage"])  # 跳过前4行说明信息
        df["voltage"] = (df["voltage"] * 1000).round(2)  # 转换为mV单位并保留两位小数
        return df
    except Exception as e:
        messagebox.showerror("错误", f"无法读取文件 {file_path}\n{e}")
        return None


# 主要的 Tkinter 类
class SignalViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("信号查看与编辑")
        self.root.geometry("1920x1080")  # 设置窗口大小
        self.root.grid_rowconfigure(0, weight=7)  # 上部信号区域占 70% 空间
        self.root.grid_rowconfigure(1, weight=2)  # 中部输入区域占 20%
        self.root.grid_rowconfigure(2, weight=1)  # 按钮区域占 10%
        self.root.grid_columnconfigure(0, weight=1)

        if not data_files:
            messagebox.showinfo("提示", "未找到任何 CSV 文件")
            self.root.quit()
            return

        # 创建 matplotlib 画布（2列×5行）
        self.fig, self.axes = plt.subplots(5, 2, figsize=(14, 6))  # 控制图像区域大小
        self.fig.tight_layout(pad=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")  # 信号显示区域

        # 创建输入框区域（10个输入框）
        input_frame = tk.Frame(root)
        input_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        tk.Label(input_frame, text="输入时间范围 (x y):").pack(anchor="w", padx=10)

        self.entries = []
        entry_container = tk.Frame(input_frame)
        entry_container.pack()

        for i in range(10):
            entry = tk.Entry(entry_container, width=10)
            entry.insert(0, "0 0")
            entry.grid(row=i // 5, column=i % 5, padx=5, pady=2)  # 5个一行，排列成 2 行
            self.entries.append(entry)

        # 按钮区域
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=2, column=0, sticky="nsew", pady=10)

        self.save_button = tk.Button(btn_frame, text="保存选定信号", command=self.save_selected_signals)
        self.save_button.pack(side=tk.LEFT, padx=20)

        self.next_button = tk.Button(btn_frame, text="下一个", command=self.load_next_file)
        self.next_button.pack(side=tk.RIGHT, padx=20)

        self.current_df = None
        self.load_file()

    # 载入 CSV 文件并更新图像
    def load_file(self):
        global current_index
        if current_index >= len(data_files):
            messagebox.showinfo("提示", "已遍历所有文件")
            return

        file_path = data_files[current_index]
        self.current_df = load_csv(file_path)
        if self.current_df is not None:
            self.update_plot()

    # 更新绘图（2列 × 5行显示）
    def update_plot(self):
        if self.current_df is None:
            return

        df = self.current_df
        total_length = df["time"].max()
        segment_length = total_length / 10

        for i, ax in enumerate(self.axes.flatten()):  # 5x2 的子图
            ax.clear()
            start_time = i * segment_length
            end_time = (i + 1) * segment_length
            segment = df[(df["time"] >= start_time) & (df["time"] < end_time)]
            ax.plot(segment["time"], segment["voltage"], label=f"片段 {i + 1}")
            ax.legend()
            ax.set_xlim(start_time, end_time)
            ax.set_ylabel("mV")

        self.axes[-1, -1].set_xlabel("时间 (s)")
        self.canvas.draw()

    # 保存选定信号
    def save_selected_signals(self):
        if self.current_df is None:
            return

        saved_segments = []
        for entry in self.entries:
            try:
                time_range = entry.get().strip()
                if time_range == "0 0":
                    continue
                start, end = map(float, time_range.split())
                if start == 0 and end == 0:
                    continue

                subset = self.current_df[(self.current_df["time"] >= start) & (self.current_df["time"] <= end)]
                if subset.empty:
                    continue

                # 生成不重复文件名
                file_index = 1
                while os.path.exists(os.path.join(path2, f"{file_index}.csv")):
                    file_index += 1

                subset.to_csv(os.path.join(path2, f"{file_index}.csv"), index=False)
                saved_segments.append(f"{time_range} -> {file_index}.csv")
            except Exception as e:
                messagebox.showerror("错误", f"无法处理输入 {entry.get()}\n{e}")

        if saved_segments:
            messagebox.showinfo("成功", "\n".join(saved_segments))

    # 载入下一个文件
    def load_next_file(self):
        global current_index
        current_index += 1
        if current_index >= len(data_files):
            messagebox.showinfo("提示", "已遍历所有文件")
            return
        self.load_file()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalViewer(root)
    root.mainloop()
