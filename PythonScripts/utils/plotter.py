from __future__ import annotations
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {"Step", "Value"}

def project_root() -> Path:
    # PythonScripts/utils/plotter.py -> PythonScripts/
    return Path(__file__).resolve().parents[1]


def default_csv_dir() -> Path:
    return project_root() / "Images" / "plots" / "csv"


def default_output_dir() -> Path:
    return project_root() / "Images" / "plots"


def load_tensorboard_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}. "
            f"Expected at least: {sorted(REQUIRED_COLUMNS)}"
        )

    df = df.sort_values("Step").reset_index(drop=True)
    return df


def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    return series.rolling(window=window, min_periods=1).mean()


def clean_name(name: str) -> str:
    # remove common TensorBoard export prefix
    name = re.sub(r"^run-\.-tag-", "", name)
    return name


def infer_title_from_filename(csv_path: Path) -> str:
    name = clean_name(csv_path.stem)
    return name.replace("_", " ").title()


def infer_ylabel_from_filename(csv_path: Path) -> str:
    name = clean_name(csv_path.stem).lower()

    if "episode_reward" in name:
        return "Episode Reward"
    if "avg_reward" in name:
        return "Average Reward"
    if "loss" in name:
        return "Loss"
    if "entropy" in name:
        return "Entropy"
    if "q1" in name or "q2" in name:
        return "Q Value"
    if "alpha" in name:
        return "Alpha"

    return "Value"


def output_png_name(csv_path: Path) -> str:
    name = clean_name(csv_path.stem)
    return f"{name}.png"


def plot_single_csv(
    csv_path: Path,
    output_dir: Path,
    smooth_window: int = 20,
    show_raw: bool = True,
    dpi: int = 220,
) -> Path:
    df = load_tensorboard_csv(csv_path)

    # Scale x-axis to 10^5 units
    x = df["Step"] / 1e5
    y = df["Value"]
    y_smooth = moving_average(y, smooth_window)

    title = infer_title_from_filename(csv_path)
    ylabel = infer_ylabel_from_filename(csv_path)

    plt.figure(figsize=(11, 6.5))

    if show_raw:
        plt.plot(
            x,
            y,
            alpha=0.25,
            linewidth=1.2,
            label="Raw",
        )

    plt.plot(
        x,
        y_smooth,
        linewidth=2.6,
        label=f"Smoothed (window={smooth_window})",
    )

    plt.title(title, fontsize=15, pad=12)
    plt.xlabel(r"Training Step ($\times 10^5$)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11)
    plt.tick_params(axis="both", labelsize=11)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_png_name(csv_path)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return out_path


def plot_all_csvs(
    csv_dir: Path,
    output_dir: Path,
    smooth_window: int = 20,
    show_raw: bool = True,
    dpi: int = 220,
) -> list[Path]:
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    saved_paths: list[Path] = []
    for csv_path in csv_files:
        out_path = plot_single_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            smooth_window=smooth_window,
            show_raw=show_raw,
            dpi=dpi,
        )
        saved_paths.append(out_path)

    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot all TensorBoard-exported CSV files from a folder."
    )

    parser.add_argument(
        "--csv-dir",
        type=str,
        default=str(default_csv_dir()),
        help="Folder containing TensorBoard CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output_dir()),
        help="Folder to save PNG plots.",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Moving average window size.",
    )

    parser.add_argument(
        "--hide-raw",
        action="store_true",
        help="Hide raw curve and only show smoothed curve.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved image DPI.",
    )

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)

    saved_paths = plot_all_csvs(
        csv_dir=csv_dir,
        output_dir=output_dir,
        smooth_window=args.smooth,
        show_raw=not args.hide_raw,
        dpi=args.dpi,
    )

    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
