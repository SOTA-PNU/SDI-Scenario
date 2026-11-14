#!/usr/bin/env python3
"""Mirror the Isaac 4.5 assets this benchmark needs to a local directory.

Only needed when the target server cannot reach the internet at run time. The
default path streams the scene from NVIDIA's S3 bucket on every run, which is
what the source host did.

Run it INSIDE the Isaac env — boto3 is already there, so nothing extra installs:

    $CARTER_WS/tools/bin/micromamba run -n isaacsim \
        python tools/mirror_assets.py /data/isaac-assets

Then point the harness at the mirror (see tools/enable_offline_assets.sh).

Mirrors 8 prefixes, about 2.3 GB — the whole dependency closure of
carter_warehouse_navigation.usd plus the robot, props and sensors it payloads.
The full Isaac 4.5 asset tree is 54 GB; you do not need it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BUCKET = "omniverse-content-production"
REGION = "us-west-2"
ROOT_KEY = "Assets/Isaac/4.5"

# Everything carter_warehouse_navigation.usd reaches, by prefix.
PREFIXES = [
    ("Isaac/Samples/ROS2/", 82.2),
    ("Isaac/Environments/Simple_Warehouse/", 538.1),
    ("Isaac/Robots/Carter/", 1596.8),
    ("Isaac/Props/Forklift/", 60.8),
    ("Isaac/Props/KLT_Bin/", 36.7),
    ("Isaac/Sensors/LeopardImaging/", 8.5),
    ("Isaac/Sensors/Slamtec/", 6.0),
    ("Isaac/Sensors/HESAI/", 1.0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest", help="local directory to mirror into")
    ap.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    args = ap.parse_args()

    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        sys.exit("boto3 not found — run this inside the Isaac env:\n"
                 "  $CARTER_WS/tools/bin/micromamba run -n isaacsim python tools/mirror_assets.py <dest>")

    dest = Path(args.dest).resolve()
    # get_assets_root_path() stats BOTH <root>/Isaac and <root>/NVIDIA before it
    # will return. An empty NVIDIA directory satisfies the second check.
    (dest / "Isaac").mkdir(parents=True, exist_ok=True)
    (dest / "NVIDIA").mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    total_files = total_bytes = skipped = 0
    print(f"mirroring s3://{BUCKET}/{ROOT_KEY}/ -> {dest}")
    print(f"expected total: {sum(mb for _, mb in PREFIXES) / 1024:.2f} GB\n")

    for prefix, approx_mb in PREFIXES:
        key_prefix = f"{ROOT_KEY}/{prefix}"
        print(f"  {prefix}  (~{approx_mb:.1f} MB)")
        n = 0
        for page in paginator.paginate(Bucket=BUCKET, Prefix=key_prefix):
            for obj in page.get("Contents", []):
                key, size = obj["Key"], obj["Size"]
                rel = key[len(ROOT_KEY) + 1:]
                target = dest / rel
                if target.is_file() and target.stat().st_size == size:
                    skipped += 1
                    continue
                if args.dry_run:
                    n += 1
                    total_files += 1
                    total_bytes += size
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET, key, str(target))
                n += 1
                total_files += 1
                total_bytes += size
                if total_files % 200 == 0:
                    print(f"      {total_files} files, {total_bytes / 1e9:.2f} GB")
        print(f"      done: {n} fetched")

    verb = "would fetch" if args.dry_run else "fetched"
    print(f"\n{verb} {total_files} files, {total_bytes / 1e9:.2f} GB "
          f"({skipped} already present and identical)")

    if not args.dry_run:
        scene = dest / "Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
        if scene.is_file():
            print(f"\nOK — the scene is present: {scene}")
            print("\nNow point the harness at this mirror:")
            print(f"    bash tools/enable_offline_assets.sh {dest}")
        else:
            print(f"\nWARNING: {scene} is missing. The mirror is incomplete.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
