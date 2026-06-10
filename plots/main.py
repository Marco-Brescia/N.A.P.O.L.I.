import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_results
from plotter import (
    plot_breaking_point_profile,
    plot_image_profile,
    plot_metric_heatmap,
    plot_perturbation_overview,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PLOT_REGISTRY = {
    "overview":       plot_perturbation_overview,
    "heatmap":        plot_metric_heatmap,
    "image_profile":  plot_image_profile,
    "breaking_point": plot_breaking_point_profile,
}

_IMAGE_FILTERED = {"image_profile", "breaking_point"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate degradation plots from pipeline result CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Root directory containing result CSVs (default: 'output').",
    )
    parser.add_argument(
        "--plots-dir", default="plots/output",
        help="Directory where PNGs are saved (default: 'plots/output').",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=list(_PLOT_REGISTRY.keys()),
        default=list(_PLOT_REGISTRY.keys()),
        metavar="PLOT_TYPE",
        help=f"Plot type(s) to generate. Choices: {', '.join(_PLOT_REGISTRY)}. Default: all.",
    )
    parser.add_argument(
        "--perturbations", nargs="+", default=None, metavar="NAME",
        help="Limit to these perturbation type names (default: all).",
    )
    parser.add_argument(
        "--images", nargs="+", default=None, metavar="IMG_ID",
        help="Limit image_profile / breaking_point plots to these image IDs "
             "(default: all).",
    )
    parser.add_argument(
        "--degradation-threshold", type=float, default=0.40, metavar="FLOAT",
        help="Relative degradation from baseline to declare a breaking point "
             "(default: 0.40 = 40%%). Applies to breaking_point plots only.",
    )
    parser.add_argument(
        "--wdp-arbiter-min", type=float, default=0.0, metavar="FLOAT",
        help="Minimum wdp relative degradation required to confirm a break "
             "(default: 0.0 = wdp must at least not improve). "
             "Applies to breaking_point plots only.",
    )
    parser.add_argument(
        "--min-consecutive", type=int, default=2, metavar="N",
        help="Consecutive degraded points required before declaring a break "
             "(default: 2). Set to 2 to avoid single-point spikes. "
             "Applies to breaking_point plots only.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Loading results from: %s", args.output_dir)
    try:
        df = load_results(args.output_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if df.empty:
        logger.warning("No data found — nothing to plot.")
        sys.exit(0)

    logger.info(
        "Loaded %d rows | %d perturbation type(s) | %d image(s) | %d category/ies",
        len(df),
        df["perturbation_type"].nunique(),
        df["img_id"].nunique(),
        df["category"].nunique(),
    )

    # -----------------------------------------------------------------------
    # Parallel plot generation
    # -----------------------------------------------------------------------
    
    # max_workers=None will use os.cpu_count() by default, which is usually a good choice for CPU-bound tasks.
    with ProcessPoolExecutor() as executor:
        futures_map = {}
        
        # Schedule plot generation tasks
        for plot_type in args.plots:
            logger.info("Scheduling '%s' plots...", plot_type)

            kwargs = dict(
                df=df,
                out_dir=args.plots_dir,
                perturbation_types=args.perturbations,
            )
            if plot_type in _IMAGE_FILTERED:
                kwargs["image_ids"] = args.images
            if plot_type == "breaking_point":
                kwargs["degradation_threshold"] = args.degradation_threshold
                kwargs["wdp_arbiter_min"]       = args.wdp_arbiter_min
                kwargs["min_consecutive"]       = args.min_consecutive

            # Submit the plot function to the executor
            future = executor.submit(_PLOT_REGISTRY[plot_type], **kwargs)
            futures_map[future] = plot_type

        # Results
        for future in as_completed(futures_map):
            plot_type = futures_map[future]
            try:
                future.result() # Exceptions will be raised here
                logger.info("Successfully completed '%s' plots.", plot_type)
            except Exception as exc:
                logger.error("Failed to generate '%s' plots: %s", plot_type, exc, exc_info=True)

    logger.info("Done. All plots saved to: %s", Path(args.plots_dir).resolve())


if __name__ == "__main__":
    main()