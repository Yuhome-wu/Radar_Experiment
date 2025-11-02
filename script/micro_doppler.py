import numpy as np
import matplotlib.pyplot as plt
import os

from read_radar_josn import parse_config

def db_to_linear(db_data):
    """将dB值转换为线性尺度"""
    return 10 ** (db_data / 10.0)

def linear_to_db(linear_data, eps=1e-12):
    """将线性尺度转换为dB值"""
    return 10 * np.log10(linear_data + eps)

def generate_md_heatmap(rd_sequence_db):
    """
    从距离-多普勒序列生成微多普勒热力图。

    参数:
        rd_sequence_db (np.ndarray): 距离-多普勒序列，单位为dB。
                                     形状为 (range_bins, velocity_bins, frames)。

    返回:
        np.ndarray: 微多普勒热力图，单位为dB。
                    形状为 (frames, velocity_bins)。
    """
    # 1. 聚合接收天线维度 (axis=3)，取最大值
    # 输入形状 (range, velocity, frames, antennas) -> (range, velocity, frames)
    rd_sequence_db_agg = np.max(rd_sequence_db, axis=3)

    # 2. 转换到线性尺度
    rd_sequence_linear = db_to_linear(rd_sequence_db_agg)

    # 3. 沿距离维度 (axis=0) 对能量求和
    # H_MD(i,j) = sum_{k=1}^{N_L} RD_i(k,j)
    # 输入形状 (range, velocity, frames)，求和后形状 (velocity, frames)
    md_linear = np.sum(rd_sequence_linear, axis=0)

    # 4. 转置以匹配 (时间, 速度) 的格式
    # (velocity, frames) -> (frames, velocity)
    md_linear_transposed = md_linear.T

    # 5. 转换回dB以便于可视化
    md_db = linear_to_db(md_linear_transposed)
    
    return md_db

def plot_md_heatmap(md_heatmap_db, vel_axis, frame_rate, output_path):
    """
    绘制并保存微多普勒热力图。

    参数:
        md_heatmap_db (np.ndarray): 微多普勒热力图 (dB)，形状 (frames, velocity_bins)。
        vel_axis (np.ndarray): 速度轴。
        frame_rate (float): 帧率 (Hz)。
        output_path (str): 图像保存路径。
    """
    num_frames = md_heatmap_db.shape[0]
    time_axis = np.arange(num_frames) / frame_rate

    # 设置支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 使用pcolormesh以正确对齐像素和坐标轴
    # 转置 md_heatmap_db 以匹配 (velocity, time) 的坐标轴
    im = ax.pcolormesh(time_axis, vel_axis, md_heatmap_db.T, cmap='jet', shading='auto')

    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('微多普勒特征热力图')

    # 归一化处理，使最大值为0dB
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('能量 (dB)')

    plt.tight_layout()
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"✅ 微多普勒热力图已保存至: {output_path}")
    plt.close(fig)

def main():
    # --- 配置路径 ---
    base_dir = r"E:\Radar_Experiment"
    rd_data_path = os.path.join(base_dir, "Outdata_npy", "range_doppler", "virtual", "rd_map_virtual_db.npy")
    vel_axis_path = os.path.join(base_dir, "Outdata_npy", "range_doppler", "virtual", "vel_axis_virtual.npy")
    output_path = os.path.join(base_dir, "micro_doppler_results", "micro_doppler_heatmap.png")
    # --- 动态配置参数 ---
    # 从JSON配置文件中动态获取帧率
    json_path = r'E:\Radar_Experiment\Data\1843_Raw_data\JSONSampleFiles\10_31.mmwave.json'
    radar_config = parse_config(json_path)
    frame_periodicity_ms = radar_config['frame_periodicity_ms']
    FRAME_RATE_HZ = 1000 / frame_periodicity_ms  # 帧周期单位是ms
    print(f"动态获取的帧率: {FRAME_RATE_HZ:.2f} Hz")
    

    # --- 数据加载 ---
    if not os.path.exists(rd_data_path):
        print(f"错误: 未找到距离-多普勒数据: {rd_data_path}")
        return
    if not os.path.exists(vel_axis_path):
        print(f"错误: 未找到速度轴数据: {vel_axis_path}")
        return

    print("正在加载距离-多普勒数据...")
    rd_sequence_db = np.load(rd_data_path)
    vel_axis = np.load(vel_axis_path)
    print("数据加载完成。")

    # --- 处理与可视化 ---
    print("正在生成微多普勒热力图...")
    md_heatmap_db = generate_md_heatmap(rd_sequence_db)
    
    # 归一化热力图，使最大值为0dB
    md_heatmap_db_normalized = md_heatmap_db - np.max(md_heatmap_db)

    print("正在绘制热力图...")
    plot_md_heatmap(md_heatmap_db_normalized, vel_axis, FRAME_RATE_HZ, output_path)

if __name__ == "__main__":
    main()