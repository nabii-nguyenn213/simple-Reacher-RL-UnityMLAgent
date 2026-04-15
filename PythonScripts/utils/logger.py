from __future__ import annotations
import os
from datetime import datetime
from typing import Any
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: str = "default",
        timestamp: bool = True,
    ):
        if timestamp:
            time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            full_log_dir = os.path.join(log_dir, f"{experiment_name}_{time_str}")
        else:
            full_log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(full_log_dir, exist_ok=True)
        self.log_dir = full_log_dir
        self.writer = SummaryWriter(log_dir=full_log_dir)

    def log_scalar(self, tag: str, value: float | int, step: int) -> None:
        self.writer.add_scalar(tag, float(value), step)

    def log_scalars(self, metrics: dict[str, Any], step: int, prefix: str | None = None) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, float(value), step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, step)

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        clean_hparams: dict[str, Any] = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        self.writer.add_hparams(clean_hparams, clean_metrics)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
