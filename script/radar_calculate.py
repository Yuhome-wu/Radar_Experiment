import json

def calc_radar_resolution(params, json_path, verbose=True):
    """计算并打印/返回雷达关键物理参数（距离与速度分辨率等）。

    参数:
        params: 由 parse_config 返回的配置字典（可不直接使用，仅为接口统一）。
        json_path: mmWave Studio 导出的 JSON 配置文件路径。

    返回:
        包含载频、斜率、有效带宽、分辨率、最大不模糊距离与速度等的字典。
    """
    with open(json_path, "r") as f:
        cfg = json.load(f)

    rf = cfg["mmWaveDevices"][0]["rfConfig"]
    prof = rf["rlProfiles"][0]["rlProfileCfg_t"]
    frame = rf["rlFrameCfg_t"]

    # ==================== 从 JSON 读取参数 ====================
    start_freq_GHz = float(prof["startFreqConst_GHz"])
    slope_MHz_per_us = float(prof["freqSlopeConst_MHz_usec"])
    ramp_end_us = float(prof["rampEndTime_usec"])
    adc_start_us = float(prof["adcStartTimeConst_usec"])
    # 兼容字段名差异
    idle_us = float(prof.get("idleTimeConst_usec", prof.get("idleTime_usec", 0.0)))
    dig_rate_ksps = float(prof["digOutSampleRate"])
    num_adc_samples = int(prof["numAdcSamples"])

    num_loops = int(frame["numLoops"])
    chirp_start = int(frame["chirpStartIdx"])
    chirp_end = int(frame["chirpEndIdx"])
    chirps_per_loop = chirp_end - chirp_start + 1
    chirps_per_frame = num_loops * chirps_per_loop

    # ==================== 常量与单位换算 ====================
    c = 3e8
    fc = start_freq_GHz * 1e9                  # Hz
    slope_Hz_per_s = slope_MHz_per_us * 1e12   # MHz/us -> Hz/s
    fs = dig_rate_ksps * 1e3                   # kS/s -> Hz

    # 时间参数
    Tramp = ramp_end_us * 1e-6                 # s
    Tadc_start = adc_start_us * 1e-6           # s
    Tc = (ramp_end_us + idle_us) * 1e-6        # 每个 chirp 周期

    # ==================== 有效带宽（基于 ADC 采样点数） ====================
    Teff = num_adc_samples / fs                # 精确有效采样时间
    B_theoretical = slope_Hz_per_s * Tramp
    B_effective = slope_Hz_per_s * Teff

    # ==================== 距离维 ====================
    range_res = c / (2.0 * B_effective)
    R_max = (fs * c) / (2.0 * slope_Hz_per_s)

    # ==================== 多普勒维 ====================
    lam = c / fc
    PRF = 1.0 / Tc
    fD_nyquist = PRF / 2.0
    v_max = lam / (4.0 * Tc)
    v_res = lam / (2.0 * chirps_per_frame * Tc)

    # ==================== 打印结果（可关闭） ====================
    if verbose:
        print("\n===== 📡 雷达配置与物理参数计算 (ADC采样精确) =====")
        print(f"载频 fc: {fc/1e9:.3f} GHz")
        print(f"频率斜率 S: {slope_MHz_per_us:.3f} MHz/us")
        print(f"理论带宽 B_theo: {B_theoretical/1e9:.3f} GHz")
        print(f"有效带宽 B_eff (ADC采样点数): {B_effective/1e9:.3f} GHz, Teff={Teff*1e6:.2f} µs")
        print(f"采样率 fs: {fs/1e6:.3f} MHz, ADC采样点数: {num_adc_samples}")
        print(f"Chirp 时长: {Tramp*1e6:.2f} µs, Idle: {idle_us:.2f} µs, 周期 Tc: {Tc*1e6:.2f} µs")
        print(f"每 Loop Chirp 数: {chirps_per_loop}, Loop 数: {num_loops}, Frame Chirp 总数: {chirps_per_frame}")
        print("-----------------------------------")
        print(f"距离分辨率 ΔR: {range_res:.4f} m")
        print(f"最大不模糊距离 R_max: {R_max:.4f} m")
        print("-----------------------------------")
        print(f"波长 λ: {lam*1e3:.3f} mm")
        print(f"PRF: {PRF:.3f} Hz, 多普勒 Nyquist: {fD_nyquist:.3f} Hz")
        print(f"速度分辨率 Δv: {v_res:.4f} m/s")
        print(f"最大不模糊速度 ±v_max: {v_max:.4f} m/s")
        print("===================================\n")

    return {
        "fc_Hz": fc,
        "slope_Hz_per_s": slope_Hz_per_s,
        "B_theoretical_Hz": B_theoretical,
        "B_effective_Hz": B_effective,
        "range_res_m": range_res,
        "R_max_m": R_max,
        "lambda_m": lam,
        "PRF_Hz": PRF,
        "v_res_m_s": v_res,
        "v_max_m_s": v_max,
        "chirps_per_loop": chirps_per_loop,
        "chirps_per_frame": chirps_per_frame,
        "num_loops": num_loops,
        "Tc_s": Tc,
        "Teff_s": Teff
    }