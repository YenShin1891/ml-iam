#!/usr/bin/env python

"""Regenerate the manuscript figures from run artifacts.

Usage:
  python scripts/plot_paper_figures.py fig3 --runs xgb_85 lstm_89 tft_95 \
      --out /root/paper_figures/fig3_scatter_CO2.png
  python scripts/plot_paper_figures.py fig4 --runs xgb_85 lstm_89 tft_95 \
      --categories C1 C2 C3 --region World \
      --out /root/paper_figures/fig4_trajectories_CO2.png
  python scripts/plot_paper_figures.py fig6 --runs xgb_85 lstm_89 tft_95 \
      --out /root/paper_figures/fig6_shap_CO2.png

fig3: IAM vs. emulated CO2 scatter per model (prints R^2 per model).
fig4: C1-C3 World CO2 trajectories + emulator-minus-IAM error row.
fig6: CO2 SHAP beeswarm per model from the saved SHAP arrays (prints the
      top-8 features per model; --panel-dir keeps the per-model panels).
Run with CUDA_VISIBLE_DEVICES="" -- nothing here needs a GPU.
"""
import argparse
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # relative metadata paths, as the dashboard resolves them

from src.visualization import paper_figures as pf  # noqa: E402

DEFAULT_RUNS = [r for r, _ in pf.RUNS]
DEFAULT_NAMES = [n for _, n in pf.RUNS]


def _add_common(p: argparse.ArgumentParser, default_out: str) -> None:
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run ids, one per panel")
    p.add_argument("--names", nargs="+", default=None, help="panel names (default: XGBoost/LSTM/TFT for the default runs)")
    p.add_argument("--target", default="Emissions|CO2")
    p.add_argument("--out", default=default_out)
    p.add_argument("--dpi", type=int, default=200)


def _names(args) -> list:
    if args.names:
        if len(args.names) != len(args.runs):
            sys.exit("--names must match --runs in length")
        return args.names
    lookup = dict(pf.RUNS)
    return [lookup.get(r, r) for r in args.runs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="figure", required=True)
    p3 = sub.add_parser("fig3", help="CO2 scatter, one panel per model")
    _add_common(p3, "paper_figures/fig3_scatter_CO2.png")
    p4 = sub.add_parser("fig4", help="C1-C3 trajectories + error row")
    _add_common(p4, "paper_figures/fig4_trajectories_CO2.png")
    p4.add_argument("--categories", nargs="+", default=["C1", "C2", "C3"])
    p4.add_argument("--region", default="World")
    p4.add_argument("--palette", choices=sorted(pf.CAT_PALETTES), default="warm")
    p4.add_argument("--x-start", type=int, default=2015)
    p6 = sub.add_parser("fig6", help="CO2 SHAP beeswarms, one panel per model")
    _add_common(p6, "paper_figures/fig6_shap_CO2.png")
    p6.set_defaults(dpi=300)
    p6.add_argument("--panel-dir", default=None, help="keep the per-model panel PNGs here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    names = _names(args)

    if args.figure == "fig3":
        r2 = pf.plot_co2_scatter(run_ids=args.runs, names=names, target=args.target, out_path=args.out, dpi=args.dpi)
        for name, v in r2.items():
            print(f"{name:8s} R^2 = {v:.3f}")
    elif args.figure == "fig6":
        top = pf.plot_shap_co2_beeswarms(
            run_ids=args.runs, names=names, target=args.target, out_path=args.out, panel_dir=args.panel_dir, dpi=args.dpi,
        )
        for name, feats in top.items():
            print(f"{name} top-{len(feats)}:")
            for k, f in enumerate(feats, 1):
                print(f"  {k}. {f}")
    else:
        counts = pf.plot_trajectories_by_category(
            run_ids=args.runs, names=names, target=args.target, categories=args.categories, region=args.region,
            palette=args.palette, out_path=args.out, x_start=args.x_start, dpi=args.dpi,
        )
        for name, n in counts.items():
            print(f"{name:8s} scenarios drawn = {n}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
