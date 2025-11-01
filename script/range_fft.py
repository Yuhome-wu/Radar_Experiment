import os
import numpy as np
from typing import Tuple, Optional

from read_radar_josn import parse_config


def next_pow2(n: int) -> int:
    """返回不小于 n 的 2 的幂。"""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def make_window(kind: str, length: int) -> np.ndarray:
    """构造长度为 length 的窗函数（1D）。支持 hann/hamming/blackman/rect。"""
    kind = (kind or "hann").lower()
    if kind in ("hann", "hanning"):
        return np.hanning(length)
    elif kind == "hamming":
        return np.hamming(length)
    elif kind == "blackman":
        return np.blackman(length)
    elif kind in ("rect", "boxcar", "none"):
        return np.ones(length, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported window type: {kind}")


def compute_range_fft(
    cube_saved: np.ndarray,
    params: dict,
    window: str = "hann",
    n_fft: Optional[int] = None,
    keep_positive: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    在采样点（快时间）维度应用窗函数并执行距离FFT，输出距离轴与频谱幅度。

    输入：
    - cube_saved：形状为 (samples, ...)，其余维度为 (chirps/loops, frames, rx/virtual_ant)
    - params：parse_config 的返回值，需包含 fs_Hz 与 slope_Hz_per_s
    - window：窗函数类型（hann/hamming/blackman/rect）
    - n_fft：FFT 点数，默认取 >= samples 的 2 次幂
    - keep_positive：是否只保留正频一半（对应物理距离 >= 0）

    返回：
    - range_axis_m：一维数组，单位 m
    - spec_mag：按正频截取后的幅度谱，形状为 (range_bins, ...)
    - range_resolution_m：距离分辨率（m），即 range_axis_m 的步长近似
    """
    c = 3e8
    fs = float(params["fs_Hz"])  # 采样率 Hz
    slope = float(params["slope_Hz_per_s"])  # 频率斜率 Hz/s

    # 维度与窗
    samples = cube_saved.shape[0]
    n_fft = n_fft or next_pow2(samples)
    w = make_window(window, samples).astype(np.float32)

    # 广播窗到采样维度
    # 将 w reshape 为 (samples, 1, 1, ...) 与 cube_saved 相乘
    reshape_dims = [samples] + [1] * (cube_saved.ndim - 1)
    w_b = w.reshape(reshape_dims)
    cube_win = cube_saved * w_b

    # 距离FFT
    spec = np.fft.fft(cube_win, n=n_fft, axis=0)

    # 频率轴（拍频）与距离轴
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)  # Hz
    if keep_positive:
        pos = freqs >= 0
        freqs = freqs[pos]
        spec = spec[pos, ...]

    # R = c * f_b / (2 * S)
    range_axis_m = (c * freqs) / (2.0 * slope)

    # 幅度（线性或 dB 可选；此处返回线性幅度）
    spec_mag = np.abs(spec)

    # 距离分辨率近似：ΔR ≈ c / (2*S) * (fs / n_fft)
    range_resolution_m = (c / (2.0 * slope)) * (fs / n_fft)
    return range_axis_m, spec_mag, range_resolution_m


def save_range_fft_outputs(
    out_dir: str,
    range_axis_m: np.ndarray,
    spec_mag: np.ndarray,
    tag: str,
):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"range_axis_{tag}.npy"), range_axis_m)
    np.save(os.path.join(out_dir, f"range_fft_{tag}.npy"), spec_mag)


def main():
    # 路径可按需修改
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    json_path = os.path.join(base_dir, r"Data\1843_Raw_data\JSONSampleFiles\10_31.mmwave.json")
    out_dir = os.path.join(base_dir, r"Outdata_npy")

    # 读取配置
    params = parse_config(json_path)

    # 加载 RAW 立方体并计算距离FFT
    raw_path = os.path.join(out_dir, "cubes", "radar_cube_raw.npy")
    if os.path.exists(raw_path):
        cube_raw = np.load(raw_path)  # (samples, chirps, frames, rx)
        range_axis_m, spec_mag, dR = compute_range_fft(cube_raw, params, window="hann")
        out_dir_fft_raw = os.path.join(out_dir, "range_fft", "raw")
        save_range_fft_outputs(out_dir_fft_raw, range_axis_m, spec_mag, tag="raw")
        print(f"✅ RAW 距离FFT完成：bins={range_axis_m.size}, ΔR≈{dR:.4f} m, 输出保存到 range_fft/raw/")
    else:
        print("⚠️ 未找到 radar_cube_raw.npy，跳过 RAW。")

    # 加载 VIRTUAL 立方体并计算距离FFT
    virt_path = os.path.join(out_dir, "cubes", "radar_cube_virtual.npy")
    if os.path.exists(virt_path):
        cube_virt = np.load(virt_path)  # (samples, loops, frames, virtual_ant)
        range_axis_m, spec_mag, dR = compute_range_fft(cube_virt, params, window="hann")
        out_dir_fft_virtual = os.path.join(out_dir, "range_fft", "virtual")
        save_range_fft_outputs(out_dir_fft_virtual, range_axis_m, spec_mag, tag="virtual")
        print(f"✅ VIRTUAL 距离FFT完成：bins={range_axis_m.size}, ΔR≈{dR:.4f} m, 输出保存到 range_fft/virtual/")
    else:
        print("⚠️ 未找到 radar_cube_virtual.npy，跳过 VIRTUAL。")


if __name__ == "__main__":
    main()