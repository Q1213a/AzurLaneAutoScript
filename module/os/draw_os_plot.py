
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

def draw(csv_path, preserve=None, target=None):
    try:
        # 读取数据 (Timestamp, AP, Coin)
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        data = np.atleast_2d(data)
        if len(data) == 0:
            print("No data in CSV")
            return
        
        # 将时间戳转换为可读的时间格式
        def format_timestamp(ts):
            return datetime.fromtimestamp(ts).strftime('%m-%d %H:%M')
            
        timestamps = data[:, 0]
        ap = data[:, 1]
        coin = data[:, 2]

        # 采样以提升性能
        if len(timestamps) > 1000:
            indices = np.linspace(0, len(timestamps) - 1, 1000).astype(int)
            timestamps = timestamps[indices]
            ap = ap[indices]
            coin = coin[indices]

        # 转换为 X 轴刻度标签
        time_labels = [format_timestamp(ts) for ts in timestamps]
        x_indices = np.arange(len(timestamps))

        fig, ax1 = plt.subplots(figsize=(12, 7))

        color_ap = 'tab:red'
        ax1.set_xlabel('Time (Month-Day Hour:Minute)')
        ax1.set_ylabel('Action Point (AP)', color=color_ap)
        ax1.plot(x_indices, ap, color=color_ap, label='AP', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_ap)
        
        # 设置 X 轴刻度
        num_ticks = 10
        tick_indices = np.linspace(0, len(x_indices) - 1, num_ticks).astype(int)
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([time_labels[i] for i in tick_indices], rotation=45)

        ax2 = ax1.twinx()
        color_coin = 'tab:blue'
        ax2.set_ylabel('Yellow Coin', color=color_coin)
        ax2.plot(x_indices, coin, color=color_coin, label='Coin', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color_coin)

        # 标记阈值
        if preserve is not None and target is not None:
            preserve = float(preserve)
            target = float(target)
            ax2.axhline(y=preserve, color='green', linestyle='--', alpha=0.6, label=f'Preserve: {preserve}')
            ax2.axhline(y=preserve + target, color='orange', linestyle='--', alpha=0.6, label=f'Threshold: {preserve + target}')
            
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('OS Simulator: AP and Coin Trend with Thresholds')
        fig.tight_layout()
        
        plot_path = csv_path.replace('.csv', '.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Drawing failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        draw(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        draw(sys.argv[1])
