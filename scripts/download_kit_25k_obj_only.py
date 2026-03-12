#!/usr/bin/env python3
"""
Download KIT ObjectModels and keep only one file per object: *_25k.obj

Pipeline:
1) Parse KIT listing page and collect mesh zip links.
2) Download each meshes.zip.
3) Extract only *_25k.obj from zip.
4) Auto-clean temporary zip files (default).

Works both locally and in Colab.
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import List, Optional, Sequence, Set, Tuple

from download_kit_objectmodels import (
    DownloadTask,
    collect_tasks,
    download_one,
    fetch_text,
    resolve_base_and_listing,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download KIT ObjectModels and keep only *_25k.obj per object."
    )
    parser.add_argument(
        "--output",
        default="data/KIT_ObjectModels_25k_obj",
        help="Final output directory for *_25k.obj files.",
    )
    parser.add_argument(
        "--project",
        default="all",
        help="KIT project split, e.g. all/SFB588/Desire/Grasp/Dexmart/Misc/SecondHands",
    )
    parser.add_argument(
        "--objects",
        default="",
        help="Optional object-name filter, comma-separated, e.g. OrangeMarmelade,BlueSaltCube",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Limit number of unique objects (0 = all matched).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers.")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=4, help="Retries per object.")
    parser.add_argument("--base-url", default="", help="Optional KIT base URL override.")
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded meshes.zip files in _tmp_zips/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_25k.obj files.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume for zip download.")
    parser.add_argument(
        "--strict-25k",
        action="store_true",
        help="Treat missing *_25k.obj in a zip as failure.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show matched objects and planned actions.")
    return parser.parse_args(argv)


def choose_25k_member(members: List[str], object_name: str) -> Optional[str]:
    candidates = [m for m in members if m.endswith("_25k.obj")]
    if not candidates:
        return None
    exact_name = f"{object_name}_25k.obj"
    for c in candidates:
        if c.endswith(exact_name):
            return c
    return candidates[0]


def process_one(
    task: DownloadTask,
    output_dir: str,
    timeout: int,
    retries: int,
    overwrite: bool,
    resume: bool,
    keep_zips: bool,
    strict_25k: bool,
) -> Tuple[str, str]:
    zip_name = f"{task.object_name}_meshes.zip"
    zip_task = replace(task, relpath=os.path.join("_tmp_zips", zip_name))

    status, zip_path = download_one(
        zip_task,
        output_dir,
        timeout=timeout,
        retries=retries,
        overwrite=overwrite,
        resume=resume,
    )
    if status == "fail":
        return "fail", zip_path

    out_obj = os.path.join(output_dir, f"{task.object_name}_25k.obj")
    if os.path.exists(out_obj) and not overwrite:
        if not keep_zips and os.path.exists(zip_path):
            os.remove(zip_path)
        return "skip", out_obj

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            member = choose_25k_member(zf.namelist(), task.object_name)
            if member is None:
                if strict_25k:
                    return "fail", f"{task.object_name}: *_25k.obj not found in zip"
                if not keep_zips and os.path.exists(zip_path):
                    os.remove(zip_path)
                return "skip", f"{task.object_name}: no *_25k.obj"
            with zf.open(member, "r") as src, open(out_obj, "wb") as dst:
                dst.write(src.read())
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{task.object_name}: unzip error: {exc}"
    finally:
        if not keep_zips and os.path.exists(zip_path):
            os.remove(zip_path)

    return "ok", out_obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    selected_objects = {x.strip().lower() for x in args.objects.split(",") if x.strip()} or None
    base_override = args.base_url.strip() or None

    base_url, listing_url = resolve_base_and_listing(args.project, base_override, args.timeout)
    page_html = fetch_text(listing_url, timeout=args.timeout)
    tasks = collect_tasks(
        page_html=page_html,
        base_url=base_url,
        selected_types={"meshes"},
        selected_objects=selected_objects,
    )

    # Keep one mesh task per object.
    dedup = {}
    for t in tasks:
        dedup[t.object_name] = t
    tasks = [dedup[k] for k in sorted(dedup.keys())]

    if args.max_objects and args.max_objects > 0:
        tasks = tasks[: args.max_objects]

    print(f"Base URL: {base_url}")
    print(f"Listing URL: {listing_url}")
    print(f"Objects matched: {len(tasks)}")
    print(f"Output dir: {os.path.abspath(args.output)}")

    if not tasks:
        print("No objects matched.")
        return 0

    if args.dry_run:
        for t in tasks[:50]:
            print(f"[DRY] {t.object_name:24s} {t.url}")
        if len(tasks) > 50:
            print(f"... ({len(tasks) - 50} more)")
        return 0

    os.makedirs(args.output, exist_ok=True)
    tmp_zip_dir = os.path.join(args.output, "_tmp_zips")
    os.makedirs(tmp_zip_dir, exist_ok=True)

    ok_count = 0
    skip_count = 0
    fail_count = 0
    failures: List[str] = []
    resume = not args.no_resume

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut_to_task = {
            ex.submit(
                process_one,
                task=t,
                output_dir=args.output,
                timeout=args.timeout,
                retries=args.retries,
                overwrite=args.overwrite,
                resume=resume,
                keep_zips=args.keep_zips,
                strict_25k=args.strict_25k,
            ): t
            for t in tasks
        }

        done = 0
        total = len(tasks)
        for fut in as_completed(fut_to_task):
            done += 1
            t = fut_to_task[fut]
            status, info = fut.result()
            if status == "ok":
                ok_count += 1
            elif status == "skip":
                skip_count += 1
            else:
                fail_count += 1
                failures.append(info)
            print(f"[{done:4d}/{total}] {status.upper():4s} {t.object_name:24s} {info}")

    # Auto-clean temp zip folder if not requested to keep.
    if not args.keep_zips and os.path.isdir(tmp_zip_dir):
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)

    print("\nSummary")
    print(f"- ok   : {ok_count}")
    print(f"- skip : {skip_count}")
    print(f"- fail : {fail_count}")
    print(f"- root : {os.path.abspath(args.output)}")

    if failures:
        print("\nFailures:")
        for msg in failures[:50]:
            print(f"- {msg}")
        if len(failures) > 50:
            print(f"... ({len(failures) - 50} more)")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
