"""
Anime Super-Resolution CLI

Unified command-line interface for all anime-sr operations.
Provides subcommands for training, inference, export, serving, GUI, validation, and benchmarking.

Usage:
    python -m anime_sr --help
    python -m anime_sr train --help
    python -m anime_sr infer --help
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Sequence

from anime_sr import __version__

# Ensure src/anime_sr is in path for internal relative imports
_src_path = str(Path(__file__).resolve().parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


# --- Global helpers -----------------------------------------------------------


def _get_log_level(verbose: bool, quiet: bool) -> str:
    """Determine log level from verbosity flags."""
    if quiet:
        return "ERROR"
    if verbose:
        return "DEBUG"
    return "INFO"


def _print_error(msg: str, suggestion: str = None) -> None:
    """Print a formatted error message with optional suggestion."""
    print(f"Error: {msg}", file=sys.stderr)
    if suggestion:
        print(f"  Hint: {suggestion}", file=sys.stderr)


def _validate_file(path: str, description: str = "File") -> Path:
    """Validate that a file exists, return Path or raise."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return p


def _validate_dir(path: str, description: str = "Directory") -> Path:
    """Validate that a directory exists, return Path or raise."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return p


# --- Subcommand handlers ------------------------------------------------------


def _handle_train(args: argparse.Namespace) -> int:
    """Handle the train subcommand."""
    import logging

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    from anime_sr.utils.config import load_config
    from anime_sr.training.orchestrator import TrainingOrchestrator, TrainingMode

    # Validate config file
    config_path = _validate_file(args.config, "Config file")

    # Load configuration
    config_dict = load_config(config_path)

    # Apply model architecture override if specified
    if args.model:
        if "model" not in config_dict:
            config_dict["model"] = {}
        config_dict["model"]["type"] = args.model

    # Apply output directory override if specified
    if args.output:
        if "paths" not in config_dict:
            config_dict["paths"] = {}
        config_dict["paths"]["checkpoint_dir"] = args.output
        config_dict["paths"]["output_dir"] = args.output

    # Create orchestrator and train
    orchestrator = TrainingOrchestrator(config_dict)

    resume_path = args.resume if args.resume else None
    orchestrator.train(resume=resume_path)

    print("Training completed successfully.")
    return 0


def _handle_infer(args: argparse.Namespace) -> int:
    """Handle the infer subcommand."""
    import logging

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    from anime_sr.inference.engine import InferenceEngine

    # Validate inputs
    input_path = _validate_file(args.input, "Input path")
    checkpoint_path = _validate_file(args.checkpoint, "Checkpoint")

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine model type
    model_type = args.model if args.model else "span"

    # Create inference engine from checkpoint
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=args.device,
    )

    # Process input (single file or directory)
    if input_path.is_file():
        # Single image
        result = engine.process_image(
            input_path=input_path,
            output_path=output_path / f"{input_path.stem}_sr{input_path.suffix}",
            scale=args.scale,
            tile_size=args.tile,
        )
        print(f"Saved: {result['output_path']}")
    else:
        # Directory of images
        from PIL import Image
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_exts])

        if not image_files:
            _print_error(f"No images found in {args.input}", "Check the input directory path")
            return 1

        print(f"Processing {len(image_files)} images...")
        for i, img_path in enumerate(image_files, 1):
            out_file = output_path / f"{img_path.stem}_sr{img_path.suffix}"
            result = engine.process_image(
                input_path=img_path,
                output_path=out_file,
                scale=args.scale,
                tile_size=args.tile,
            )
            if args.verbose:
                print(f"  [{i}/{len(image_files)}] {img_path.name} -> {out_file.name}")

        print(f"Processed {len(image_files)} images. Output: {output_path}")

    return 0


def _handle_export(args: argparse.Namespace) -> int:
    """Handle the export subcommand."""
    import logging

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    from anime_sr.inference.engine import InferenceEngine

    # Validate checkpoint
    checkpoint_path = _validate_file(args.checkpoint, "Checkpoint")

    # Validate format
    valid_formats = ["onnx", "torchscript"]
    if args.format not in valid_formats:
        _print_error(
            f"Invalid format: {args.format}",
            f"Choose from: {', '.join(valid_formats)}"
        )
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine model type
    model_type = args.model if args.model else "span"

    # Create engine and export
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )

    # Export based on format
    if args.format == "onnx":
        engine.export_onnx(output_path)
    elif args.format == "torchscript":
        engine.export_torchscript(output_path)

    print(f"Exported to: {output_path}")
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    """Handle the serve subcommand."""
    import logging

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    # Load config if provided
    config_dict = {}
    if args.config:
        from anime_sr.utils.config import load_config
        config_path = _validate_file(args.config, "Config file")
        config_dict = load_config(config_path)

    try:
        import uvicorn
    except ImportError:
        _print_error(
            "uvicorn is required for serving",
            "Install with: pip install uvicorn fastapi"
        )
        return 1

    from anime_sr.api.inference_api import create_app

    # Create app with config
    app = create_app(config=config_dict)

    print(f"Starting server on {args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def _handle_gui(args: argparse.Namespace) -> int:
    """Handle the gui subcommand."""
    # Load config if provided
    config_dict = {}
    if args.config:
        from anime_sr.utils.config import load_config
        config_path = _validate_file(args.config, "Config file")
        config_dict = load_config(config_path)

    try:
        from anime_sr.gui import launch_gui
    except ImportError:
        _print_error(
            "GUI dependencies not installed",
            "Install with: pip install -e '.[gui]'"
        )
        return 1

    launch_gui(config=config_dict)
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    """Handle the validate subcommand."""
    import logging

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    from anime_sr.inference.engine import InferenceEngine

    # Validate checkpoint
    checkpoint_path = _validate_file(args.checkpoint, "Checkpoint")

    # Determine model type
    model_type = args.model if args.model else "span"

    # Create engine (this validates the checkpoint can be loaded)
    try:
        engine = InferenceEngine.from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
        )
        print(f"Checkpoint is valid: {checkpoint_path}")
        print(f"Model type: {model_type}")
        print(f"Model parameters: {sum(p.numel() for p in engine.model.parameters()):,}")
        return 0
    except Exception as e:
        _print_error(
            f"Checkpoint validation failed: {e}",
            "Ensure the checkpoint matches the specified model architecture"
        )
        return 1


def _handle_benchmark(args: argparse.Namespace) -> int:
    """Handle the benchmark subcommand."""
    import logging
    import time

    log_level = _get_log_level(args.verbose, args.quiet)
    logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s: %(message)s")

    import numpy as np
    from PIL import Image

    from anime_sr.inference.engine import InferenceEngine

    # Validate checkpoint
    checkpoint_path = _validate_file(args.checkpoint, "Checkpoint")

    # Validate input image if provided
    input_path = None
    if args.input:
        input_path = _validate_file(args.input, "Input image")

    # Determine model type
    model_type = args.model if args.model else "span"

    # Create engine
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )

    # Prepare test image
    if input_path:
        img = np.array(Image.open(input_path).convert("RGB"))
    else:
        # Create a synthetic test image
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({args.iterations} iterations)...")
    for _ in range(min(5, args.iterations)):
        _ = engine.run(img)

    # Benchmark
    print(f"Benchmarking ({args.iterations} iterations)...")
    times = []
    for i in range(args.iterations):
        start = time.perf_counter()
        _ = engine.run(img)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        if args.verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.iterations}] {elapsed*1000:.1f} ms")

    # Report results
    times_arr = np.array(times)
    print(f"\nBenchmark Results ({args.iterations} iterations):")
    print(f"  Mean:   {times_arr.mean()*1000:.1f} ms")
    print(f"  Std:    {times_arr.std()*1000:.1f} ms")
    print(f"  Min:    {times_arr.min()*1000:.1f} ms")
    print(f"  Max:    {times_arr.max()*1000:.1f} ms")
    print(f"  Median: {np.median(times_arr)*1000:.1f} ms")

    return 0


# --- Argument parser construction ---------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="anime-sr",
        description="Anime Super-Resolution with Progressive Ensemble Distillation",
        epilog="Run 'python -m anime_sr <command> --help' for more information on a command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Global config file path (can be overridden by subcommand --config)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- Train --------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train a super-resolution model",
        description="Train a super-resolution model using the specified configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr train --config configs/train.yaml
  python -m anime_sr train --config configs/train.yaml --model span --output runs/exp1
  python -m anime_sr train --config configs/train.yaml --resume checkpoints/last.pth
  python -m anime_sr train --config configs/train.yaml --verbose
""",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        choices=["span", "mamba_pan", "ensemble"],
        default=None,
        help="Model architecture (overrides config)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    train_parser.set_defaults(func=_handle_train)

    # -- Infer --------------------------------------------------------------
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on images",
        description="Run super-resolution inference on images or directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr infer --input image.png --output output/ --checkpoint model.pth
  python -m anime_sr infer --input images/ --output output/ --checkpoint model.pth --scale 4
  python -m anime_sr infer --input image.png --output output/ --checkpoint model.pth --tile 256
  python -m anime_sr infer --input image.png --output output/ --checkpoint model.pth --tta
  python -m anime_sr infer --input image.png --output output/ --checkpoint model.pth --device cpu
""",
    )
    infer_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image file or directory",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path (file or directory)",
    )
    infer_parser.add_argument(
        "--model",
        type=str,
        choices=["span", "mamba_pan", "ensemble"],
        default=None,
        help="Model architecture (default: span)",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    infer_parser.add_argument(
        "--scale",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="Upscale factor (default: 4)",
    )
    infer_parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tile size for tiled inference, 0 for full image (default: 0)",
    )
    infer_parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation for better quality",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )
    infer_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    infer_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    infer_parser.set_defaults(func=_handle_infer)

    # --- Export -------------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to ONNX or TorchScript",
        description="Export a trained model to ONNX or TorchScript format for deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr export --checkpoint model.pth --format onnx --output model.onnx
  python -m anime_sr export --checkpoint model.pth --format torchscript --output model.pt
  python -m anime_sr export --checkpoint model.pth --format onnx --output model.onnx --model span
""",
    )
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        choices=["span", "mamba_pan", "ensemble"],
        default=None,
        help="Model architecture (default: span)",
    )
    export_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    export_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    export_parser.set_defaults(func=_handle_export)

    # -- Serve --------------------------------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the inference API server",
        description="Start a REST API server for super-resolution inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr serve --config configs/api.yaml
  python -m anime_sr serve --host 127.0.0.1 --port 8080
  python -m anime_sr serve --config configs/api.yaml --reload
""",
    )
    serve_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to API config YAML",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes",
    )
    serve_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    serve_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    serve_parser.set_defaults(func=_handle_serve)

    # --- GUI ----------------------------------------------------------------
    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the desktop GUI",
        description="Launch the graphical user interface for super-resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr gui
  python -m anime_sr gui --config configs/gui.yaml
""",
    )
    gui_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to GUI config YAML",
    )
    gui_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    gui_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    gui_parser.set_defaults(func=_handle_gui)

    # --- Validate -----------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a model checkpoint",
        description="Validate that a checkpoint can be loaded and is compatible with the specified architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr validate --checkpoint model.pth
  python -m anime_sr validate --checkpoint model.pth --model mamba_pan
""",
    )
    validate_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to validate",
    )
    validate_parser.add_argument(
        "--model",
        type=str,
        choices=["span", "mamba_pan", "ensemble"],
        default=None,
        help="Model architecture (default: span)",
    )
    validate_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    validate_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    validate_parser.set_defaults(func=_handle_validate)

    # --- Benchmark ----------------------------------------------------------
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model performance",
        description="Benchmark inference speed of a model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m anime_sr benchmark --checkpoint model.pth
  python -m anime_sr benchmark --checkpoint model.pth --input test.png --iterations 50
  python -m anime_sr benchmark --checkpoint model.pth --model span --iterations 100
""",
    )
    benchmark_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    benchmark_parser.add_argument(
        "--model",
        type=str,
        choices=["span", "mamba_pan", "ensemble"],
        default=None,
        help="Model architecture (default: span)",
    )
    benchmark_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Test image path (uses synthetic image if not provided)",
    )
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    benchmark_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    benchmark_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    benchmark_parser.set_defaults(func=_handle_benchmark)

    return parser


# --- Main entry point ---------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for the anime-sr CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to appropriate handler
    try:
        return args.func(args)
    except FileNotFoundError as e:
        _print_error(str(e), "Check the file path and try again")
        return 1
    except ImportError as e:
        _print_error(
            f"Missing dependency: {e}",
            "Install required dependencies with: pip install -e '.[all]'"
        )
        return 1
    except RuntimeError as e:
        _print_error(str(e))
        return 1
    except Exception as e:
        _print_error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
