"""Offline D-1B local fixture CLI: no native or inference integration."""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
from .field import export_bundle, import_png8, radial_airbrush


def main() -> None:
    parser = argparse.ArgumentParser(description="D-1B local soft-guidance fixture authoring")
    parser.add_argument("prefix", type=Path, help="output basename (.u16le/.png/.json)")
    parser.add_argument("--source-png", type=Path, help="import PNG8 G/GA/RGB/RGBA as scalar guidance")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--center-x", type=float, default=32)
    parser.add_argument("--center-y", type=float, default=32)
    parser.add_argument("--radius", type=float, default=27)
    parser.add_argument("--hardness", type=float, default=0.3)
    parser.add_argument("--gradient-length", type=float, default=0.9)
    parser.add_argument("--density", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=570)
    args = parser.parse_args()
    origin_hash = None
    if args.source_png:
        content = args.source_png.read_bytes()
        field = import_png8(content)
        origin_hash = sha256(content).hexdigest()
    else:
        field = radial_airbrush(args.width, args.height, center_x=args.center_x,
                                center_y=args.center_y, radius_cells=args.radius,
                                hardness=args.hardness, gradient_length=args.gradient_length)
    authoring = None if args.source_png else {
        "center_x": args.center_x, "center_y": args.center_y,
        "radius_cells": args.radius, "hardness": args.hardness,
        "gradient_length": args.gradient_length,
    }
    meta = export_bundle(args.prefix, field, seed=args.seed,
                         centre_density=args.density, source_png_sha256=origin_hash,
                         authoring_parameters=authoring)
    manifest_sha = sha256(args.prefix.with_suffix(".json").read_bytes()).hexdigest()
    print(f"scalar_sha256={meta['values']['sha256']} preview_sha256={meta['display']['sha256']} manifest_sha256={manifest_sha}")


if __name__ == "__main__":
    main()
