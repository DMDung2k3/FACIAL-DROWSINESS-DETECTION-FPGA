import argparse
import time
import random
import os
import logging
from typing import Tuple, List, Optional
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import psutil
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# NVML để đo công suất GPU
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
except ImportError:
    nvmlInit = None  # Không đo công suất nếu thiếu pynvml

# deterministic import của simps
try:
    from scipy.integrate import simpson as simps
except ImportError:
    def simps(y, x):
        return np.trapz(y, x)

from dataset.datasets import WLFWDatasets
from models.pfld import PFLDInference

# ------------------ Setup Logging ------------------
def setup_logging(log_file: str = "evaluation.log"):
    """Thiết lập logging để ghi kết quả."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

# ------------------ Determinism & CuDNN ------------------
def set_seed(seed: int = 42):
    """Đặt seed để đảm bảo tính deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = True

# ------------------ Metrics ------------------
def compute_nme(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Tính Normalized Mean Error (NME)."""
    N, L2, _ = preds.shape
    L = L2
    rmse = np.zeros(N)
    for i in range(N):
        pts_pred, pts_gt = preds[i], target[i]
        if L == 19:
            inter = 34
        elif L == 29:
            inter = np决心.norm(pts_gt[8] - pts_gt[9])
        elif L == 68:
            inter = np.linalg.norm(pts_gt[36] - pts_gt[45])
        elif L == 98:
            inter = np.linalg.norm(pts_gt[60] - pts_gt[72])
        else:
            raise ValueError(f"Wrong number of landmarks: {L}")
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (inter * L)
    return rmse

def compute_auc(errors: np.ndarray, failure_threshold: float, step: float = 0.0001, show_curve: bool = False) -> Tuple[float, float]:
    """Tính AUC và Failure Rate."""
    n = len(errors)
    x = np.arange(0.0, failure_threshold + step, step)
    ced = [(errors <= xi).sum() / n for xi in x]
    auc = simps(ced, x) / failure_threshold
    fr = 1.0 - ced[-1]
    if show_curve:
        plt.plot(x, ced)
        plt.xlabel("Error")
        plt.ylabel("Cumulative Distribution")
        plt.title("CED Curve")
        plt.grid(True)
        plt.show()
    return auc, fr

# ------------------ Evaluation Utilities ------------------
def measure_power(device: torch.device, handle, before: bool = True) -> Optional[float]:
    """Đo công suất tiêu thụ (W) cho GPU hoặc ước lượng cho CPU."""
    if device.type == "cuda" and handle is not None:
        return nvmlDeviceGetPowerUsage(handle) / 1000.0  # Chuyển từ mW sang W
    elif device.type == "cpu":
        tdp = 65  # Giả sử TDP của CPU là 65W (có thể tùy chỉnh)
        cpu_usage = psutil.cpu_percent(interval=None) / 100.0
        return tdp * cpu_usage  # Ước lượng công suất CPU
    return None

def warm_up(model: torch.nn.Module, device: torch.device, input_shape: Tuple[int, ...] = (1, 3, 112, 112)):
    """Chạy warm-up để ổn định thời gian suy luận."""
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

# ------------------ Evaluation ------------------
def validate(dataloader: DataLoader, model: torch.nn.Module, device: torch.device, num_images: int = 10) -> None:
    """Đánh giá mô hình trên tập dữ liệu."""
    model.eval()
    proc = psutil.Process(os.getpid())
    inf_times, cpu_utils, nmes, power_vals = [], [], [], []

    # Khởi tạo NVML nếu dùng GPU
    handle = None
    if device.type == "cuda" and nvmlInit is not None:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

    # Warm-up
    warm_up(model, device)

    # Chọn ngẫu nhiên mẫu
    total = len(dataloader.dataset)
    indices = random.sample(range(total), min(num_images, total))
    subset = Subset(dataloader.dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    with torch.no_grad():
        for img, gt, _, _ in loader:
            img = img.to(device)
            gt_np = gt.reshape(1, -1, 2).cpu().numpy()

            # Đo CPU timing
            cpu_before = proc.cpu_times().user + proc.cpu_times().system

            # Đo công suất trước inference
            p_before = measure_power(device, handle, before=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            _, lm = model(img)

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

            # Đo CPU sau inference
            cpu_after = proc.cpu_times().user + proc.cpu_times().system
            inf_times.append(dt * 1000)  # ms
            cpu_utils.append(((cpu_after - cpu_before) / dt) * 100)

            # Đo công suất sau inference
            p_after = measure_power(device, handle, before=False)
            if p_before is not None and p_after is not None:
                power_vals.append((p_before + p_after) / 2)

            lm_np = lm.cpu().numpy().reshape(1, -1, 2)
            nmes.extend(compute_nme(lm_np, gt_np).tolist())

    # Tính metrics
    avg_inf_ms = np.mean(inf_times)
    avg_inf_s = avg_inf_ms / 1000.0
    avg_cpu = np.mean(cpu_utils)
    avg_nme = np.mean(nmes)
    fps = len(indices) / (sum(inf_times) / 1000.0)
    avg_power = np.mean(power_vals) if power_vals else None
    fom = avg_power * avg_inf_s if avg_power is not None else None
    mem_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else proc.memory_info().rss / (1024**2)
    )
    auc, fr = compute_auc(np.array(nmes), 0.1)

    # Ghi log và in kết quả
    metrics = {
        "Device": device.type.upper(),
        "Power Consumption (W)": f"{avg_power:.2f}" if avg_power else "N/A",
        "Inference Time (ms)": f"{avg_inf_ms:.2f}",
        "FOM (W×s)": f"{fom:.4f}" if fom else "N/A",
        "Memory Footprint (MB)": f"{mem_mb:.2f}",
        "FPS (img/s)": f"{fps:.2f}",
        "Average CPU Usage (%)": f"{avg_cpu:.2f}",
        "Average Test Accuracy (NME)": f"{avg_nme:.4f}",
        "AUC @0.1 failureThresh": f"{auc:.4f}",
        "Failure Rate": f"{fr:.4f}",
    }
    logging.info("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        logging.info(f"{k:25}: {v}")
    logging.info("==========================\n")

# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate PFLD model on landmark detection task.")
    parser.add_argument('--model_path', type=str, default='./checkpoint/snapshot/checkpoint.pth.tar', help="Path to model checkpoint")
    parser.add_argument('--test_dataset', type=str, default='./data/test_data/list.txt', help="Path to test dataset")
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help="Device to run evaluation")
    parser.add_argument('--num_images', type=int, default=50, help="Number of images to evaluate")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--log_file', type=str, default='evaluation.log', help="Path to log file")
    args = parser.parse_args()

    # Thiết lập logging
    setup_logging(args.log_file)

    # Kiểm tra thiết bị
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.error("CUDA is not available! Falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)

    # Điều chỉnh num_workers cho CPU
    if args.device == 'cpu':
        args.num_workers = min(args.num_workers, 2)  # Giảm tải cho CPU
        logging.info(f"Adjusted num_workers to {args.num_workers} for CPU.")

    # Đặt seed
    set_seed(args.seed)

    # Tải mô hình
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        model = PFLDInference().to(device)
        model.load_state_dict(ckpt['pfld_backbone'])
        logging.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # Tải dữ liệu
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        ds = WLFWDatasets(args.test_dataset, transform)
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == 'cuda'),
        )
        logging.info(f"Loaded dataset from {args.test_dataset}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Đánh giá
    validate(dl, model, device, num_images=args.num_images)

if __name__ == "__main__":
    main()