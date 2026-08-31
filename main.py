#!/usr/bin/env python3
"""
Oil Spill Detection & Vessel Attribution System — Entry Point

Usage:
    # Run the demo pipeline (no data required)
    python main.py --demo

    # Run the web dashboard
    python main.py --web

    # Run on a specific image
    python main.py --image path/to/sar_image.tif

    # Run with custom parameters
    python main.py --demo --lat 19.0 --lon 72.8 --hindcast 48
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_demo(args):
    """Run the demo pipeline."""
    from src.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

    config = PipelineConfig(
        hindcast_hours=args.hindcast,
        forecast_hours=args.forecast,
        n_particles=args.particles,
        pixel_size_m=args.pixel_size,
        output_dir=args.output,
    )

    pipeline = PipelineOrchestrator(config)
    result = pipeline.run_demo(
        center_lat=args.lat,
        center_lon=args.lon,
        detection_time=args.time,
    )

    # Print report
    if result.attribution:
        from src.attribution.vessel_scorer import VesselScorer
        scorer = VesselScorer()
        print(scorer.format_report(result.attribution))

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for key, value in result.summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    if result.errors:
        print("\nERRORS:")
        for e in result.errors:
            print(f"  - {e}")

    print(f"\nResult saved to: {args.output}/")
    return result


def run_web(args):
    """Run the web dashboard."""
    from src.web.app import create_app

    app = create_app()
    print(f"\n🛢️  Oil Spill Detection Dashboard")
    print(f"   Running at http://{args.host}:{args.port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=args.debug)


def run_image(args):
    """Run detection on a specific image."""
    from src.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig

    config = PipelineConfig(
        pixel_size_m=args.pixel_size,
        output_dir=args.output,
    )

    pipeline = PipelineOrchestrator(config)
    result = pipeline.run(
        image_path=args.image,
        detection_time=args.time,
    )

    print(f"\nDetection complete: {result.detection.num_slicks} slick(s) found")
    for i, gp in enumerate(result.geometric_properties):
        print(f"  Slick {i}: area={gp.area_km2:.4f} km², "
              f"compactness={gp.compactness:.3f}, "
              f"elongation={gp.elongation:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Oil Spill Detection & Vessel Attribution System"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo = subparsers.add_parser("demo", help="Run demo pipeline with synthetic data")
    demo.add_argument("--lat", type=float, default=20.5, help="Center latitude")
    demo.add_argument("--lon", type=float, default=60.3, help="Center longitude")
    demo.add_argument("--time", type=str, default="2025-01-01T12:00:00",
                       help="Detection time (ISO-8601)")
    demo.add_argument("--hindcast", type=float, default=36, help="Hindcast hours")
    demo.add_argument("--forecast", type=float, default=48, help="Forecast hours")
    demo.add_argument("--particles", type=int, default=200, help="Number of particles")
    demo.add_argument("--pixel-size", type=float, default=10.0, help="Pixel size in metres")
    demo.add_argument("--output", type=str, default="./output", help="Output directory")

    # Web command
    web = subparsers.add_parser("web", help="Run the web dashboard")
    web.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    web.add_argument("--port", type=int, default=5000, help="Port")
    web.add_argument("--debug", action="store_true", help="Debug mode")

    # Image command
    img = subparsers.add_parser("detect", help="Detect oil spills in an image")
    img.add_argument("image", type=str, help="Path to SAR/EO image")
    img.add_argument("--time", type=str, default=None, help="Image time (ISO-8601)")
    img.add_argument("--pixel-size", type=float, default=10.0, help="Pixel size in metres")
    img.add_argument("--output", type=str, default="./output", help="Output directory")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "demo":
        run_demo(args)
    elif args.command == "web":
        run_web(args)
    elif args.command == "detect":
        run_image(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
