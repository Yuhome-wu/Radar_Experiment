import os
import numpy as np
from typing import Optional, Tuple

from read_radar_josn import parse_config
from radar_calculate import calc_radar_resolution


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def make_window(kind: str, length: int) -> np.ndarray:
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


def mag_to_db(mag: np.ndarray, normalize: bool = True, eps: float = 1e-12) -> np.ndarray:
    """将线性幅度转换为 dB。
    - normalize=True 时，先按全局最大值归一化到 [0,1]，最大值对应 0 dB。
    - eps 防止 log(0)。
    """
    if normalize:
        maxv = float(np.max(mag))
        if maxv > 0:
            mag = mag / maxv
    return 20.0 * np.log10(mag + eps)


def range_fft_complex(
    cube_saved: np.ndarray,
    params: dict,
    window: str = "hann",
    n_fft: Optional[int] = None,
    keep_positive: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """对采样维度做加窗与距离FFT，保留复数结果并返回物理距离轴。"""
    c = 3e8
    fs = float(params["fs_Hz"])          # 采样率 Hz
    slope = float(params["slope_Hz_per_s"])  # 频率斜率 Hz/s

    samples = cube_saved.shape[0]
    n_fft = n_fft or next_pow2(samples)
    w = make_window(window, samples).astype(np.float32)
    w_b = w.reshape([samples] + [1] * (cube_saved.ndim - 1))

    cube_win = cube_saved * w_b
    spec = np.fft.fft(cube_win, n=n_fft, axis=0)

    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)  # Hz
    if keep_positive:
        pos = freqs >= 0
        freqs = freqs[pos]
        spec = spec[pos, ...]

    range_axis_m = (c * freqs) / (2.0 * slope)
    return range_axis_m, spec


def doppler_fft(
    spec_range: np.ndarray,
    prf_hz: float,
    window: str = "hann",
    n_fft: Optional[int] = None,
    axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在慢时间维（loops/有效脉冲序列）做多普勒FFT，返回速度轴与复数谱（已fftshift）。
    spec_range 形状应为 (..., slow_time_len, ...)，axis 指出慢时间所在维度（virtual: axis=1）。
    """
    lam = float(calc_params_cached.get("lambda_m"))  # 从缓存读取波长
    if lam is None:
        raise RuntimeError("lambda_m not available; ensure calc_radar_resolution was called.")

    slow_len = spec_range.shape[axis]
    n_fft = n_fft or next_pow2(slow_len)
    w = make_window(window, slow_len).astype(np.float32)

    # 将窗函数广播到指定轴
    shape = [1] * spec_range.ndim
    shape[axis] = slow_len
    w_b = w.reshape(shape)

    spec_doppler = np.fft.fft(spec_range * w_b, n=n_fft, axis=axis)
    spec_doppler = np.fft.fftshift(spec_doppler, axes=axis)

    # 速度轴：fD 经过 fftshift，对应 -PRF/2..+PRF/2
    fD = np.fft.fftfreq(n_fft, d=1.0 / prf_hz)
    fD = np.fft.fftshift(fD)
    v_axis = (lam / 2.0) * fD

    return v_axis, spec_doppler


def compute_rd_virtual(
    cube_virtual_saved: np.ndarray,
    params: dict,
    prf_hz: float,
    range_window: str = "hann",
    range_nfft: Optional[int] = None,
    doppler_window: str = "hann",
    doppler_nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算虚阵列数据的 Range-Doppler 图。返回 (range_axis_m, v_axis_m_s, rd_mag)。"""
    # 距离FFT（复数）
    range_axis_m, spec_r = range_fft_complex(
        cube_virtual_saved, params, window=range_window, n_fft=range_nfft, keep_positive=True
    )

    # 多普勒FFT沿 loops 维（虚阵列保存为 (samples, loops, frames, virtual_ant) -> rangeFFT 为 (bins, loops, frames, virtual_ant)）
    v_axis_m_s, spec_d = doppler_fft(spec_r, prf_hz, window=doppler_window, n_fft=doppler_nfft, axis=1)

    rd_mag = np.abs(spec_d)
    return range_axis_m, v_axis_m_s, rd_mag


def compute_rd_raw_select_tx(
    cube_raw_saved: np.ndarray,
    params: dict,
    prf_hz: float,
    tx_id: int = 0,
    range_window: str = "hann",
    range_nfft: Optional[int] = None,
    doppler_window: str = "hann",
    doppler_nfft: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 RAW 数据（(samples, chirps, frames, rx)）选择某个 TX，在 loops 维上做多普勒FFT。
    返回 (range_axis_m, v_axis_m_s, rd_mag)。
    """
    # 距离FFT（复数）
    range_axis_m, spec_r = range_fft_complex(
        cube_raw_saved, params, window=range_window, n_fft=range_nfft, keep_positive=True
    )

    # 将 chirps 维拆成 (num_loops, chirps_per_loop)，并选取指定 TX 对应的 chirp 位置
    chirps_per_frame = int(params["chirps_per_frame"])  # = num_loops * chirps_per_loop
    num_loops = int(params["num_loops"])
    chirps_per_loop = int(params["chirps_per_loop"])
    assert chirps_per_frame == num_loops * chirps_per_loop

    # 从 parse_config 获取每个位置对应的 TX ID 序列
    tx_id_per_pos = params.get("tx_id_per_chirp_pos")
    if not tx_id_per_pos or len(tx_id_per_pos) != chirps_per_loop:
        raise ValueError("tx_id_per_chirp_pos 缺失或长度不匹配，无法从 RAW 选择 TX 进行多普勒FFT。")

    try:
        tx_pos = tx_id_per_pos.index(int(tx_id))
    except ValueError:
        raise ValueError(f"指定的 TX ID {tx_id} 不在 tx_id_per_chirp_pos={tx_id_per_pos} 中。")

    # spec_r 形状: (bins, chirps_per_frame, frames, rx)
    bins, chirps_total, frames, rx = spec_r.shape
    assert chirps_total == chirps_per_frame

    # 先 reshape 到 (bins, num_loops, chirps_per_loop, frames, rx)，再选择指定 tx_pos
    spec_r = spec_r.reshape(bins, num_loops, chirps_per_loop, frames, rx)
    spec_r_tx = spec_r[:, :, tx_pos, :, :]  # (bins, num_loops, frames, rx)

    # 多普勒FFT沿 num_loops 维（axis=1）
    v_axis_m_s, spec_d = doppler_fft(spec_r_tx, prf_hz, window=doppler_window, n_fft=doppler_nfft, axis=1)
    rd_mag = np.abs(spec_d)

    return range_axis_m, v_axis_m_s, rd_mag


def save_rd_outputs(out_dir: str, range_axis_m: np.ndarray, v_axis_m_s: np.ndarray, rd_mag: np.ndarray, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"range_axis_{tag}.npy"), range_axis_m)
    np.save(os.path.join(out_dir, f"vel_axis_{tag}.npy"), v_axis_m_s)
    np.save(os.path.join(out_dir, f"rd_map_{tag}.npy"), rd_mag)


def save_rd_outputs_db(out_dir: str, range_axis_m: np.ndarray, v_axis_m_s: np.ndarray, rd_mag: np.ndarray, tag: str):
    """保存 dB 版本的 RD 图（归一化到最大值）。"""
    os.makedirs(out_dir, exist_ok=True)
    rd_db = mag_to_db(rd_mag, normalize=True)
    np.save(os.path.join(out_dir, f"range_axis_{tag}.npy"), range_axis_m)
    np.save(os.path.join(out_dir, f"vel_axis_{tag}.npy"), v_axis_m_s)
    np.save(os.path.join(out_dir, f"rd_map_{tag}_db.npy"), rd_db)


def save_rd_outputs_per_rx(out_dir: str, rd_mag: np.ndarray, base_tag: str):
    """将 RD 图按 RX 天线拆分保存。
    rd_mag 形状应为 (R_bins, V_bins, frames, rx)。
    保存为 rd_map_{base_tag}_rx{idx}.npy。
    """
    os.makedirs(out_dir, exist_ok=True)
    rx_dir = os.path.join(out_dir, "rx")
    os.makedirs(rx_dir, exist_ok=True)
    if rd_mag.ndim != 4:
        raise ValueError(f"rd_mag 维度应为4，得到 {rd_mag.shape}")
    rx = rd_mag.shape[-1]
    for rxi in range(rx):
        fname = os.path.join(rx_dir, f"rd_map_{base_tag}_rx{rxi}.npy")
        np.save(fname, rd_mag[..., rxi])
    # 额外提示文件保存位置
    print(f"   └─ 已按 RX 拆分保存到：rd_map_{base_tag}_rx*.npy")


def save_rd_outputs_per_rx_db(out_dir: str, rd_mag: np.ndarray, base_tag: str):
    """将 RD 图按 RX 天线拆分保存 dB 版本，使用全局最大值归一化。"""
    os.makedirs(out_dir, exist_ok=True)
    rx_dir = os.path.join(out_dir, "rx")
    os.makedirs(rx_dir, exist_ok=True)
    if rd_mag.ndim != 4:
        raise ValueError(f"rd_mag 维度应为4，得到 {rd_mag.shape}")
    rx = rd_mag.shape[-1]
    maxv = float(np.max(rd_mag))
    eps = 1e-12
    for rxi in range(rx):
        slice_lin = rd_mag[..., rxi]
        # 先按全局最大值归一化，再转 dB
        if maxv > 0:
            slice_lin = slice_lin / maxv
        slice_db = 20.0 * np.log10(slice_lin + eps)
        fname = os.path.join(rx_dir, f"rd_map_{base_tag}_rx{rxi}_db.npy")
        np.save(fname, slice_db)
    print(f"   └─ 已按 RX 拆分保存到：rd_map_{base_tag}_rx*_db.npy")


# 供 doppler_fft 使用的缓存（lambda_m）
calc_params_cached = {}


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    json_path = os.path.join(base_dir, r"Data\1843_Raw_data\JSONSampleFiles\10_31.mmwave.json")
    out_dir = os.path.join(base_dir, r"Outdata_npy")

    # 读取配置与物理参数（用于 PRF、λ）
    params = parse_config(json_path)
    calc = calc_radar_resolution(params, json_path, verbose=False)
    prf_hz = float(calc["PRF_Hz"])  # 1/Tc
    calc_params_cached["lambda_m"] = float(calc["lambda_m"])  # 供 doppler_fft 使用

    # 处理 VIRTUAL（推荐用于多普勒）
    virt_path = os.path.join(out_dir, "cubes", "radar_cube_virtual.npy")
    if os.path.exists(virt_path):
        cube_virt = np.load(virt_path)  # (samples, loops, frames, virtual_ant)
        r_axis, v_axis, rd_mag = compute_rd_virtual(
            cube_virt, params, prf_hz, range_window="hann", range_nfft=None, doppler_window="hann", doppler_nfft=None
        )
        # 输出到分类目录：Outdata_npy/range_doppler/virtual
        out_dir_virtual = os.path.join(out_dir, "range_doppler", "virtual")
        # 仅保存 dB 版本的 RD 图与轴文件
        save_rd_outputs_db(out_dir_virtual, r_axis, v_axis, rd_mag, tag="virtual")
        print(f"✅ VIRTUAL Range-Doppler 完成（仅 dB）：R_bins={r_axis.size}, V_bins={v_axis.size}, 保存到 range_doppler/virtual/")
    else:
        print("⚠️ 未找到 radar_cube_virtual.npy，跳过 VIRTUAL。")

    # 处理 RAW（按需选择 TX，且按 RX 拆分保存）
    raw_path = os.path.join(out_dir, "cubes", "radar_cube_raw.npy")
    if os.path.exists(raw_path):
        cube_raw = np.load(raw_path)  # (samples, chirps, frames, rx)
        # 自动获取可用 TX 列表（优先来自 tx_id_per_chirp_pos；否则来自 tx_enable_mask；再否则按 [0..num_tx-1]）
        tx_ids = None
        tx_id_per_pos = params.get("tx_id_per_chirp_pos")
        if tx_id_per_pos and len(tx_id_per_pos) > 0:
            tx_ids = sorted(set(int(x) for x in tx_id_per_pos))
        if not tx_ids:
            mask = int(params.get("tx_enable_mask", 0))
            if mask:
                tx_ids = [i for i in range(8) if (mask >> i) & 1]
        if not tx_ids:
            num_tx = int(params.get("num_tx", 3))
            tx_ids = list(range(num_tx))

        for tx in tx_ids:
            r_axis, v_axis, rd_mag = compute_rd_raw_select_tx(
                cube_raw, params, prf_hz, tx_id=tx, range_window="hann", range_nfft=None, doppler_window="hann", doppler_nfft=None
            )
            tag = f"raw_tx{tx}"
            # 输出到分类目录：Outdata_npy/range_doppler/raw/tx{tx}/ 和 rx/
            out_dir_tx = os.path.join(out_dir, "range_doppler", "raw", f"tx{tx}")
            # 仅保存 dB 版本（包含整体与按 RX 拆分）
            save_rd_outputs_db(out_dir_tx, r_axis, v_axis, rd_mag, tag=tag)
            save_rd_outputs_per_rx_db(out_dir_tx, rd_mag, base_tag=tag)
            print(f"✅ RAW(TX{tx}) Range-Doppler 完成（仅 dB）：R_bins={r_axis.size}, V_bins={v_axis.size}, 保存到 range_doppler/raw/tx{tx}/")
    else:
        print("⚠️ 未找到 radar_cube_raw.npy，跳过 RAW。")


if __name__ == "__main__":
    main()