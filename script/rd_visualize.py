import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def pick_cmap(name: str = None):
    if name:
        try:
            return plt.get_cmap(name)
        except Exception:
            pass
    for cand in ("turbo", "jet", "viridis"):
        try:
            return plt.get_cmap(cand)
        except Exception:
            continue
    return plt.cm.viridis


def load_virtual(base_dir: str, rx: int | None = None):
    vdir = os.path.join(base_dir, "Outdata_npy", "range_doppler", "virtual")
    r_axis = np.load(os.path.join(vdir, "range_axis_virtual.npy"))
    v_axis = np.load(os.path.join(vdir, "vel_axis_virtual.npy"))
    rd_db = np.load(os.path.join(vdir, "rd_map_virtual_db.npy"))  # (R, V, frames) or (R, V, frames, rx)
    if rd_db.ndim == 4:
        if rx is None:
            # 聚合 RX（取最大值）
            rd_db = np.max(rd_db, axis=-1)
        else:
            rd_db = rd_db[..., rx]
    return r_axis, v_axis, rd_db


def load_raw(base_dir: str, tx: int, rx: int | None):
    tdir = os.path.join(base_dir, "Outdata_npy", "range_doppler", "raw", f"tx{tx}")
    r_axis = np.load(os.path.join(tdir, f"range_axis_raw_tx{tx}.npy"))
    v_axis = np.load(os.path.join(tdir, f"vel_axis_raw_tx{tx}.npy"))
    if rx is None:
        rd_db = np.load(os.path.join(tdir, f"rd_map_raw_tx{tx}_db.npy"))  # (R, V, frames) or (R, V, frames, rx)
        if rd_db.ndim == 4:
            # 聚合 RX（取最大值）
            rd_db = np.max(rd_db, axis=-1)
    else:
        rd_db = np.load(os.path.join(tdir, "rx", f"rd_map_raw_tx{tx}_rx{rx}_db.npy"))  # (R, V, frames)
    return r_axis, v_axis, rd_db


def plot_rd(r_axis: np.ndarray, v_axis: np.ndarray, rd_db: np.ndarray, frame: int = 0,
            vmin: float = -60.0, vmax: float = 0.0, title: str = "Range-Doppler (dB)", cmap: str | None = None,
            interactive: bool = False, save_path: str | None = None):
    assert rd_db.ndim == 3, f"期望 rd_db 维度为 (R, V, frames)，实际为 {rd_db.shape}"
    R, V, F = rd_db.shape
    frame = max(0, min(frame, F - 1))

    extent = [float(v_axis.min()), float(v_axis.max()), float(r_axis.min()), float(r_axis.max())]
    cm = pick_cmap(cmap)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rd_db[:, :, frame], origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Magnitude (dB)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title(title + f" | frame={frame}")
    plt.tight_layout()

    if interactive and F > 1:
        # 仅保留帧滑块；移除颜色范围（vmin/vmax）的交互
        axframe = plt.axes([0.15, 0.02, 0.7, 0.03])
        sframe = Slider(axframe, "frame", 0, F - 1, valinit=frame, valstep=1)

        def on_change_frame(val):
            f = int(sframe.val)
            im.set_data(rd_db[:, :, f])
            ax.set_title(title + f" | frame={f}")
            fig.canvas.draw_idle()

        sframe.on_changed(on_change_frame)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def save_all_frames(r_axis: np.ndarray, v_axis: np.ndarray, rd_db: np.ndarray,
                    out_dir: str, base_name: str,
                    vmin: float = -60.0, vmax: float = 0.0, cmap: str | None = None):
    """按帧批量保存 RD 图为 PNG。
    - base_name 用于文件名前缀，例如 'rd_virtual_agg' 或 'rd_virtual_rx2'。
    - 输出文件命名：{base_name}_fXXXX.png
    - 保存完成后会在控制台打印提示（总帧数与输出目录）。
    """
    assert rd_db.ndim == 3, f"期望 rd_db 维度为 (R, V, frames)，实际为 {rd_db.shape}"
    R, V, F = rd_db.shape
    extent = [float(v_axis.min()), float(v_axis.max()), float(r_axis.min()), float(r_axis.max())]
    cm = pick_cmap(cmap)
    os.makedirs(out_dir, exist_ok=True)

    for f in range(F):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rd_db[:, :, f], origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Magnitude (dB)")
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Range (m)")
        ax.set_title(base_name + f" | frame={f}")
        plt.tight_layout()
        fname = os.path.join(out_dir, f"{base_name}_f{f:04d}.png")
        plt.savefig(fname, dpi=200)
        plt.close(fig)

def export_raw_rd_all_frames(tx: int,
                             base_dir: str | None = None,
                             out_dir: str | None = None):
    """专用 RAW 可视化导出函数：RX 聚合、默认颜色范围、自动批量保存。
    参数：
    - tx: TX 编号（如 0/1/2）
    - base_dir: 项目根目录；默认取脚本上级目录
    - out_dir: 输出目录；默认保存到 Outdata_npy/vis/rd_raw_tx{tx}_agg

    行为：
    - 读取 Outdata_npy/range_doppler/raw/tx{tx}/ 下的 dB 数据与轴文件
    - 始终对 RX 维度做聚合（取最大值），得到 (R, V, frames)
    - 始终使用默认颜色范围（vmin=-60, vmax=0）与默认配色（优先 turbo）
    - 自动批量保存所有帧为 PNG（命名 rd_raw_tx{tx}_agg_fXXXX.png）
    """
    base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # RX 聚合：传入 rx=None，让 load_raw 在需要时对 RX 做最大值聚合
    r_axis, v_axis, rd_db = load_raw(base_dir, tx, rx=None)
    file_prefix = f"rd_raw_tx{tx}_agg"
    out_dir = out_dir or os.path.join(base_dir, "Outdata_npy", "vis", file_prefix)

    default_vmin = -60.0
    default_vmax = 0.0
    default_cmap = None  # pick_cmap(None) -> 优先 turbo

    save_all_frames(r_axis, v_axis, rd_db, out_dir, file_prefix,
                    vmin=default_vmin, vmax=default_vmax, cmap=default_cmap)
    print(f"[RD] RAW 导出完成：TX{tx} (RX-aggregated) -> {out_dir}")


def main():
    """主入口
    不接受任何命令行参数：
    - 始终聚合所有虚拟 RX，并批量保存所有帧到固定目录（rd_virtual_agg）。
    - 另外也会对 RAW 的 TX0/1/2 进行 RX 聚合并批量保存到各自目录（rd_raw_tx{tx}_agg）。
    - 若某个 TX 的 RAW 数据缺失，会跳过并打印提示。
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # 始终聚合所有虚拟 RX（取最大值）
    r_axis, v_axis, rd_db = load_virtual(base_dir, rx=None)
    file_prefix = "rd_virtual_agg"
    out_dir = os.path.join(base_dir, "Outdata_npy", "vis", file_prefix)

    # 固定颜色范围与配色
    default_vmin = -60.0
    default_vmax = 0.0
    default_cmap = None  # pick_cmap(None) -> 优先 turbo

    # 始终批量保存所有帧到固定目录
    save_all_frames(r_axis, v_axis, rd_db, out_dir, file_prefix, vmin=default_vmin, vmax=default_vmax, cmap=default_cmap)

    # 注：主流程结束，输出目录提示
    print(f"[RD] Virtual 导出完成： -> {out_dir}")

    # 同步导出 RAW（RX 聚合），覆盖 TX0/1/2；缺失则跳过
    for tx in (0, 1, 2):
        try:
            export_raw_rd_all_frames(tx=tx, base_dir=base_dir, out_dir=None)
        except FileNotFoundError as e:
            print(f"[RD] RAW 数据缺失：TX{tx}，已跳过。详情：{e}")
        except Exception as e:
            print(f"[RD] RAW 导出失败：TX{tx}，错误：{e}")


if __name__ == "__main__":
    main()