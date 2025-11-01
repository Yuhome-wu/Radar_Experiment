import os
import numpy as np
from read_radar_josn import parse_config
from radar_calculate import calc_radar_resolution

def parse_bin_to_cube(bin_path, params, dtype=np.int16):
    """根据配置解析BIN文件，并在尺寸不匹配时智能推断维度。

    目标形状: (frames, chirps_per_frame, rx, samples, IQ=2)
    其中 IQ=2 表示 I/Q 各占一个 int16。
    """
    samples = int(params["num_adc_samples"])
    rx = int(params["num_rx"])
    frames_cfg = int(params["num_frames"])
    chirps_cfg = int(params["chirps_per_frame"])

    # 读取整个文件（使用内存映射以降低峰值内存占用）
    data = np.memmap(bin_path, dtype=dtype, mode="r")
    actual_count = data.size  # 以 int16 元素计数
    units_per_chirp = rx * samples * 2  # 每个 chirp 的元素数（含 I/Q）

    # 优先假定帧数正确，推断每帧 chirp 数
    frames_use = frames_cfg
    chirps_use = None
    if frames_use > 0 and actual_count % (frames_use * units_per_chirp) == 0:
        chirps_use = actual_count // (frames_use * units_per_chirp)
    else:
        chirps_use = None

    # 若无法整除，尝试使用配置的 chirps 推断帧数
    if chirps_use is None or chirps_use <= 0:
        if chirps_cfg > 0 and actual_count % (chirps_cfg * units_per_chirp) == 0:
            frames_use = actual_count // (chirps_cfg * units_per_chirp)
            chirps_use = chirps_cfg
        else:
            # 仍无法匹配，提供详细错误信息
            raise ValueError(
                f"Raw file element count={actual_count} mismatch with cfg: "
                f"frames={frames_cfg}, chirps={chirps_cfg}, rx={rx}, samples={samples}; "
                f"units_per_chirp={units_per_chirp}"
            )

    expected_count = frames_use * chirps_use * units_per_chirp
    if expected_count != actual_count:
        # 再次校正，尽量用整除关系修复
        frames_chirps_total = actual_count // units_per_chirp
        if frames_use > 0 and frames_chirps_total % frames_use == 0:
            chirps_use = frames_chirps_total // frames_use
        elif chirps_use > 0 and frames_chirps_total % chirps_use == 0:
            frames_use = frames_chirps_total // chirps_use
        else:
            raise ValueError("Unable to reconcile raw file size with configuration.")

    # 形状重建（memmap reshape 不会复制数据）
    data = data.reshape((frames_use, chirps_use, rx, samples, 2))
    i = data[..., 0].astype(np.float32)
    q = data[..., 1].astype(np.float32)
    cube = i + 1j * q

    # 回写推断结果，保证后续流程维度一致
    params["num_frames"] = frames_use
    params["chirps_per_frame"] = chirps_use
    num_loops = int(params.get("num_loops", 1))
    if num_loops > 0:
        params["chirps_per_loop"] = chirps_use // num_loops

    return cube

def _build_tx_positions(params, chirps_per_loop):
    """根据配置构建每个 loop 内用于虚阵列重排的 chirp 位置列表。

    语义说明：返回的是“chirp 的位置索引列表”（0..chirps_per_loop-1），用于从一个 loop 的 block 中抽取对应 TX 的数据，顺序决定虚拟阵列中 TX 的排列。

    优先级：
    1) 若提供 params['tx_order']（TX ID 顺序，如 [0,2,1]），则根据 params['tx_id_per_chirp_pos'] 将 TX ID 映射为对应的 chirp 位置，返回位置列表。
    2) 若无 tx_order，但提供 params['tx_id_per_chirp_pos']，则采用“按 chirp 顺序”的位置列表 [0..chirps_per_loop-1]（对应 TX 顺序为 tx_id_per_chirp_pos）。
    3) 若仅有 tx_enable_mask（整数或列表），则按掩码的置位顺序生成 TX ID，再在可用 chirp 中按顺序筛选对应位置。
    4) 兜底：返回 [0..min(chirps_per_loop, num_tx)-1]。
    """
    # 可能的 TX-ID 映射（每个 chirp 位置对应的 TX ID），来源于 parse_config 的 rlChirps 解析
    tx_id_per_pos = params.get("tx_id_per_chirp_pos")

    # 1) 显式 TX ID 顺序 -> 位置列表
    if "tx_order" in params and params["tx_order"] is not None and tx_id_per_pos:
        tx_order = params["tx_order"]
        if isinstance(tx_order, (list, tuple)) and all(isinstance(x, (int, np.integer)) for x in tx_order):
            positions = []
            for tx_id in tx_order:
                # 在一个 loop 的序列中查找该 TX ID 所对应的 chirp 位置
                found_pos = None
                for pos, tid in enumerate(tx_id_per_pos):
                    if tid == int(tx_id):
                        found_pos = pos
                        break
                if found_pos is None:
                    raise ValueError(f"TX ID {tx_id} not present in tx_id_per_chirp_pos={tx_id_per_pos}")
                positions.append(found_pos)
            return positions

    # 2) 无 tx_order，但有 tx_id_per_chirp_pos，按 chirp 序（即采集时序）返回所有位置
    if tx_id_per_pos and len(tx_id_per_pos) == chirps_per_loop:
        return list(range(chirps_per_loop))

    # 3) 仅有 tx_enable_mask：将掩码的置位按升序转换为 TX ID 列表，再按 chirp 顺序选择对应位置
    if "tx_enable_mask" in params and params["tx_enable_mask"] is not None:
        mask = params["tx_enable_mask"]
        if isinstance(mask, int):
            tx_ids = []
            bit = 0
            while (1 << bit) <= mask:
                if mask & (1 << bit):
                    tx_ids.append(bit)
                bit += 1
            if tx_ids:
                if tx_id_per_pos:
                    positions = [pos for pos, tid in enumerate(tx_id_per_pos) if tid in tx_ids]
                    if positions:
                        return positions
                # 没有 tx_id_per_pos，按 [0..min(chirps_per_loop, len(tx_ids))-1]
                return list(range(min(chirps_per_loop, len(tx_ids))))
        elif isinstance(mask, (list, tuple)):
            tx_ids = [int(x) for x in mask if isinstance(x, (int, np.integer))]
            if tx_ids:
                if tx_id_per_pos:
                    positions = [pos for pos, tid in enumerate(tx_id_per_pos) if tid in tx_ids]
                    if positions:
                        return positions
                return list(range(min(chirps_per_loop, len(tx_ids))))

    # 4) 兜底：按最常见的 TDM 轮发顺序取前 min(chirps_per_loop, num_tx) 个位置
    num_tx = int(params.get("num_tx", chirps_per_loop))
    return list(range(min(chirps_per_loop, num_tx)))


def cube_to_virtual(cube, params):
    """重排为虚天线格式（TDM-MIMO）。

    输入：
    - cube：形状 (frames, chirps_per_frame, rx, samples)
    - params：需包含
      - num_loops：每帧的 doppler 循环数（一个帧包含多少个连续的循环）
      - num_tx：配置的 TX 数量（用于默认顺序推断）
      - num_rx：实际 RX 通道数量
      - 可选 tx_order（TX ID 顺序，如 [0,2,1]）/tx_enable_mask/tx_id_per_chirp_pos（由 JSON 解析的 chirp->TX 映射）

    重排结果：
    - (frames, num_loops, num_virtual_ant, samples)，其中 num_virtual_ant = num_tx_used * num_rx

    关系说明：
    - chirps_per_frame 必须能被 num_loops 整除：chirps_per_loop = chirps_per_frame // num_loops
    - 若提供 tx_order（TX ID 顺序），则根据 tx_id_per_chirp_pos 映射为位置并确定 num_tx_used；否则默认 num_tx_used = min(chirps_per_loop, num_tx)
    """
    frames, chirps_per_frame, rx, samples = cube.shape
    num_loops = int(params["num_loops"])
    num_tx = int(params["num_tx"])
    num_rx = int(params["num_rx"])

    if chirps_per_frame % num_loops != 0:
        raise ValueError(
            f"chirps_per_frame={chirps_per_frame} not divisible by num_loops={num_loops}."
        )
    chirps_per_loop = chirps_per_frame // num_loops

    # 根据配置构建每个 loop 内使用的 TX 所对应的 chirp 位置
    tx_positions = _build_tx_positions(params, chirps_per_loop)
    num_tx_used = len(tx_positions)

    cube = cube.reshape((frames, num_loops, chirps_per_loop, rx, samples))
    num_virtual = num_tx_used * num_rx
    virtual_cube = np.zeros((frames, num_loops, num_virtual, samples), dtype=np.complex64)

    for f in range(frames):
        for l in range(num_loops):
            block = cube[f, l]  # 形状: (chirps_per_loop, rx, samples)
            virt_list = []
            # 根据 tx_positions 抽取对应的 chirp -> TX 数据
            for pos in tx_positions:
                if pos < 0 or pos >= chirps_per_loop:
                    raise IndexError(f"tx_position {pos} out of range for chirps_per_loop={chirps_per_loop}")
                for rxi in range(num_rx):
                    virt_list.append(block[pos, rxi, :])
            virtual_cube[f, l, :, :] = np.stack(virt_list, axis=0)

    return virtual_cube

def main():
    json_path = r"E:\Radar_Experriment\Data\1843_Raw_data\JSONSampleFiles\10_31.mmwave.json"   
    bin_path = r"E:\Radar_Experriment\Data\1843_Raw_data\10_31\adc_data_fall_2_Raw_0.bin"       
    out_dir = r"E:\Radar_Experriment\Outdata_npy"

    params = parse_config(json_path)
    calc_radar_resolution(params, json_path, verbose=True)
    
    cube_raw = parse_bin_to_cube(bin_path, params)  # 形状: (frames, chirps, rx, samples)
    # 保存时重排为 (samples, chirps, frames, rx)
    cube_raw_saved = np.transpose(cube_raw, (3, 1, 0, 2))
    cubes_dir = os.path.join(out_dir, "cubes")
    os.makedirs(cubes_dir, exist_ok=True)
    np.save(os.path.join(cubes_dir, "radar_cube_raw.npy"), cube_raw_saved)
    print("✅ 已保存 cubes/radar_cube_raw.npy 形状:", cube_raw_saved.shape)

    cube_virtual = cube_to_virtual(cube_raw, params)
    # 保存为 (samples, loops, frames, virtual_ant)
    cube_virtual_saved = np.transpose(cube_virtual, (3, 1, 0, 2))
    np.save(os.path.join(cubes_dir, "radar_cube_virtual.npy"), cube_virtual_saved)
    print("✅ 已保存 cubes/radar_cube_virtual.npy 形状:", cube_virtual_saved.shape)

if __name__ == "__main__":
    main()