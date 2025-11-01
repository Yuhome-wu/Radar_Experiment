"""
read_radar_josn.py
-------------------
封装 mmWave Studio 导出的 JSON 配置读取函数，供各模块复用。

使用示例：
    from read_radar_josn import parse_config
    params = parse_config(json_path)
    print(params["num_frames"], params["chirps_per_frame"], params["num_rx"]) 
"""

import json
from typing import List

def _count_bits(hex_str: str) -> int:
    """计算启用天线数量，例如 '0xF' -> 4"""
    return bin(int(hex_str, 16)).count("1")

def parse_config(json_path: str) -> dict:
    """从 mmWaveStudio JSON 配置中解析关键参数。

    返回字典包含以下键：
        - num_rx: 接收天线数
        - num_tx: 发射天线数
        - rx_enable_mask: RX 使能掩码（整数）
        - tx_enable_mask: TX 使能掩码（整数）
        - num_adc_samples: 每个 chirp 的 ADC 采样点数
        - num_loops: 每帧的 doppler 循环数（一个帧包含多少个连续的循环）
        - num_frames: 帧数
        - chirps_per_loop: 每个 loop 内的 chirp 数
        - chirps_per_frame: 每帧总 chirp 数 = num_loops * chirps_per_loop
        - iqSwapSel, chInterleave: 原始数据格式相关标志
    """
    with open(json_path, "r") as f:
        cfg = json.load(f)

    dev = cfg["mmWaveDevices"][0]
    rf = dev["rfConfig"]
    frame = rf["rlFrameCfg_t"]
    profile = rf["rlProfiles"][0]["rlProfileCfg_t"]
    rl_chirps = rf.get("rlChirps", [])

    # 基础参数
    rx_en_hex = rf["rlChanCfg_t"]["rxChannelEn"]
    tx_en_hex = rf["rlChanCfg_t"]["txChannelEn"]
    num_rx = _count_bits(rx_en_hex)
    num_tx = _count_bits(tx_en_hex)
    rx_enable_mask = int(rx_en_hex, 16)
    tx_enable_mask = int(tx_en_hex, 16)
    num_adc_samples = int(profile["numAdcSamples"])
    num_loops = int(frame["numLoops"])
    num_frames = int(frame["numFrames"])

    # 以 FrameCfg 的 chirpStartIdx/chirpEndIdx 计算每 loop 的 chirp 数，
    # 更贴近实际采集范围（避免 rlChirps 列表包含未用于的 chirp 定义）。
    chirp_start = int(frame["chirpStartIdx"])
    chirp_end = int(frame["chirpEndIdx"]) 
    chirps_per_loop = chirp_end - chirp_start + 1
    chirps_per_frame = num_loops * chirps_per_loop

    # 从 rlChirps 提取每个 chirp 的 TX 使能（用于构建 TX-CHIRP 映射顺序）
    # 先构建索引到掩码的映射（支持范围定义）
    chirp_tx_mask_map = {}
    for entry in rl_chirps:
        cfg_chirp = entry.get("rlChirpCfg_t", {})
        s_idx = int(cfg_chirp.get("chirpStartIdx", chirp_start))
        e_idx = int(cfg_chirp.get("chirpEndIdx", s_idx))
        tx_en_hex = cfg_chirp.get("txEnable", "0x0")
        tx_mask = int(tx_en_hex, 16)
        for idx in range(s_idx, e_idx + 1):
            chirp_tx_mask_map[idx] = tx_mask

    # 基于 Frame 的 chirpStartIdx..chirpEndIdx，按顺序构建每个 chirp 位置所对应的 TX ID
    # 对于 TDM-MIMO，通常每个 chirp 仅有一个 TX 置位；若有多个置位，取最低位（LSB）对应的 TX ID 作为近似。
    def _lowest_set_bit(mask: int) -> int:
        if mask <= 0:
            return 0
        bit = 0
        while mask:
            if mask & 1:
                return bit
            mask >>= 1
            bit += 1
        return 0

    tx_id_per_chirp_pos: List[int] = []
    for idx in range(chirp_start, chirp_end + 1):
        tx_mask = chirp_tx_mask_map.get(idx, 0)
        tx_id = _lowest_set_bit(tx_mask)
        tx_id_per_chirp_pos.append(tx_id)

    # IQ 格式相关（按需使用）
    iqSwapSel = dev["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["iqSwapSel"]
    chInterleave = dev["rawDataCaptureConfig"]["rlDevDataFmtCfg_t"]["chInterleave"]

    # 采样率与斜率（用于物理映射）
    start_freq_GHz = float(profile["startFreqConst_GHz"])  # GHz
    slope_MHz_per_us = float(profile["freqSlopeConst_MHz_usec"])  # MHz/us
    fs_ksps = float(profile["digOutSampleRate"])  # kS/s
    fs_Hz = fs_ksps * 1e3
    slope_Hz_per_s = slope_MHz_per_us * 1e12

    return {
        "num_rx": num_rx,
        "num_tx": num_tx,
        "rx_enable_mask": rx_enable_mask,
        "tx_enable_mask": tx_enable_mask,
        "num_adc_samples": num_adc_samples,
        "num_loops": num_loops,
        "num_frames": num_frames,
        "chirps_per_loop": chirps_per_loop,
        "chirps_per_frame": chirps_per_frame,
        "start_freq_GHz": start_freq_GHz,
        "fs_Hz": fs_Hz,
        "slope_Hz_per_s": slope_Hz_per_s,
        "tx_id_per_chirp_pos": tx_id_per_chirp_pos,
        "iqSwapSel": iqSwapSel,
        "chInterleave": chInterleave,
    }

__all__ = ["parse_config"]