"""Command-line entry point: ``python -m lunacore register LABEL.xml -o OUT``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .errors import LunaCoreError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lunacore", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register", help="register a PDS4 product onto an LRO base map")
    r.add_argument("label", type=Path, help="Chandrayaan-2 PDS4 .xml label")
    r.add_argument("-o", "--out", type=Path, required=True, help="output directory")
    r.add_argument("--reference", type=Path, help="local reference GeoTIFF (skips WMS)")
    r.add_argument("--config", type=Path, help="JSON file with PipelineConfig sections")
    r.add_argument("--gsd", type=float, help="source GSD in m/px if the label lacks it")
    r.add_argument("--wms-url", help="WMS endpoint override")
    r.add_argument("--layer", help="WMS layer override")
    r.add_argument("--matcher", choices=("loftr", "sift"), help="feature matcher")
    r.add_argument("--device", help="torch device: auto, cpu, cuda, mps")
    r.add_argument("--prior", choices=("auto", "corners", "bbox", "scale"),
                   help="geometric prior used to pre-align the source")
    r.add_argument("--ecc", action="store_true",
                   help="enable ECC dense refinement (for similarly lit pairs)")
    r.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = (PipelineConfig.from_dict(json.loads(args.config.read_text()))
           if args.config else PipelineConfig())
    if args.wms_url:
        cfg.wms.url = args.wms_url
    if args.layer:
        cfg.wms.layer = args.layer
    if args.matcher:
        cfg.matching.matcher = args.matcher
    if args.device:
        cfg.matching.device = args.device
    if args.prior:
        cfg.harmonization.prior = args.prior
    if args.ecc:
        cfg.geometry.ecc_refine = True

    from .pipeline import register

    try:
        result = register(args.label, args.out, cfg, reference_path=args.reference,
                          gsd_override_m=args.gsd)
    except LunaCoreError as exc:
        logging.getLogger("lunacore").error("registration failed: %s", exc)
        return 2
    json.dump({"registered": str(result.registered_path), "report": str(result.report_path),
               "metrics": result.metrics}, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0
