"""Command-line entry point for the proposed GNSS-constrained SLAM method."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from .datasets import canonical_name, make_dataset
from .errors import DatasetError
from .profiles import profile

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


@app.command()
def slam(
    dataset: str = typer.Argument(..., help="Dataset name: rtk-slam, m2dgr, or i2nav-robot"),
    root: Path = typer.Argument(..., help="Dataset root directory, or one sequence ZIP/bag"),
    sequence: Optional[str] = typer.Option(None, "--sequence", "-s", help="Sequence name"),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output root"),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="YAML overrides"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Interactive KISS-SLAM viewer"),
    n_scans: int = typer.Option(-1, "--n-scans", "-n", help="Frame limit; -1 runs all frames"),
    jump: int = typer.Option(0, "--jump", "-j", help="First frame index"),
    refuse_scans: bool = typer.Option(False, "--refuse-scans", help="Build final occupancy map"),
):
    """Run reliability-gated GNSS correction on a supported recorded dataset."""
    try:
        canonical = canonical_name(dataset)
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        geodesy = {
            "quality_policy": "covariance_only", "timestamp_offset_s": 0.0,
            "map_translation_m": [0.0, 0.0, 0.0], "yaw_alignment_rad": 0.0,
        }
        gnss = profile(canonical)
        runtime = {
            "out_dir": str(output), "gnss": gnss,
            "odometry": {"preprocessing": {"max_range": 50.0, "min_range": 1.0, "deskew": True}},
        }
        if config:
            supplied = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            runtime = _merge(runtime, supplied)
            geodesy = _merge(geodesy, supplied.get("geodesy", {}))
        geodesy.update({
            "set_lidar_roll_pitch": runtime["gnss"].get("set_lidar_roll_pitch", False),
            "lidar_roll_deg": runtime["gnss"].get("lidar_roll_deg", 0.0),
            "lidar_pitch_deg": runtime["gnss"].get("lidar_pitch_deg", 0.0),
        })
        ds, canonical, resolved_sequence, input_path = make_dataset(
            canonical, root, sequence, geodesy, output / ".cache"
        )
        runtime_path = output / f"{canonical}-{resolved_sequence}-config.yaml"
        runtime.pop("geodesy", None)
        runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
        typer.echo(f"Dataset: {canonical}")
        typer.echo(f"Sequence: {resolved_sequence}")
        typer.echo(f"Input: {input_path}")
        typer.echo(f"Output root: {output}")
        from kiss_slam.pipeline import SlamPipeline
        SlamPipeline(
            dataset=ds, config_file=runtime_path, visualize=visualize,
            n_scans=n_scans, jump=jump, refuse_scans=refuse_scans,
        ).run().print()
    except DatasetError as exc:
        typer.echo(f"Dataset error: {exc}", err=True)
        raise typer.Exit(2) from exc


def run():
    app()


if __name__ == "__main__":
    run()

