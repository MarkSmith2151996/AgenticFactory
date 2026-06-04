from __future__ import annotations

import asyncio
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agentic_factory.executor import execute_llm_agent, execute_service_agent
from agentic_factory.loader import get_agent


KEEPA_URL = "https://keepa.com/#!finder/%7B%22f%22%3A%7B%22BUY_BOX_SHIPPING_current%22%3A%7B%22filterType%22%3A%22number%22%2C%22type%22%3A%22inRange%22%2C%22filter%22%3A22.56%2C%22filterTo%22%3A108.72%7D%2C%22SALES_current%22%3A%7B%22filterType%22%3A%22number%22%2C%22type%22%3A%22lessThanOrEqual%22%2C%22filter%22%3A68534%2C%22filterTo%22%3Anull%7D%2C%22COUNT_NEW_FBA_current%22%3A%7B%22filterType%22%3A%22number%22%2C%22type%22%3A%22inRange%22%2C%22filter%22%3A2%2C%22filterTo%22%3A10%7D%2C%22salesRankDrops90%22%3A%7B%22filterType%22%3A%22number%22%2C%22type%22%3A%22greaterThanOrEqual%22%2C%22filter%22%3A40%2C%22filterTo%22%3Anull%7D%2C%22sellerIdsFBA%22%3A%7B%22filterType%22%3A%22text%22%2C%22type%22%3A%22isOneOf%22%2C%22filter%22%3A%22AQ9D9YX2TEHKA%2C%20A3GDUY63NULHCX%2C%20A1UWLDVGZSXGKG%2C%20A2LYX7FKX6AP16%2C%20A1GD4LR25P9QFR%2C%20A2FQL10UCAZUR8%2C%20A37KLPQG0QTZGU%2C%20A3DMYIN9K0RYXC%2C%20A86M66Z7Z0Z95%2C%20A34ABF00R5C25H%22%7D%2C%22AMAZON_outOfStock%22%3A%7B%22filterType%22%3A%22boolean%22%2C%22filter%22%3A%22on%22%2C%22type%22%3A%22equals%22%7D%7D%2C%22t%22%3A%22f%22%2C%22s%22%3A%5B%7B%22colId%22%3A%22SALES_current%22%2C%22sort%22%3A%22asc%22%7D%2C%7B%22colId%22%3A%22monthlySold%22%2C%22sort%22%3A%22desc%22%7D%5D%7D"
LOCAL_CSV_DIR = Path("/tmp/keepa-test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live YAML agent pipeline E2E test.")
    parser.add_argument(
        "--csv-path",
        action="append",
        default=[],
        help="Existing Keepa CSV path to feed into data-runner; skips keepa-downloader.",
    )
    return parser.parse_args()


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def extract_csv_paths(download_output: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for item in download_output.get("files", []):
        if not isinstance(item, dict):
            continue
        path = item.get("file") or item.get("path") or item.get("csv_path") or item.get("output_path")
        if path:
            paths.append(str(path))
    return paths


def localize_csv_paths(csv_paths: list[str]) -> list[str]:
    localized = []
    for path in csv_paths:
        localized.append(localize_csv_path(path))
    return localized


def localize_csv_path(csv_path: str) -> str:
    if Path(csv_path).exists():
        return csv_path
    if not csv_path.startswith("/Users/"):
        print(f"CSV path is not local and not a Mac /Users path, leaving unchanged: {csv_path}")
        return csv_path

    LOCAL_CSV_DIR.mkdir(parents=True, exist_ok=True)
    destination = LOCAL_CSV_DIR / Path(csv_path).name
    if destination.exists():
        print(f"Using cached local CSV: {destination}")
        return str(destination)

    filename = Path(csv_path).name
    url = f"http://anthonys-macbook-pro:8095/files/{filename}"
    curl_result = subprocess.run(
        ["curl", "-fsSL", "-o", str(destination), url],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if curl_result.returncode == 0 and destination.exists():
        print(f"Downloaded CSV from keepa service file endpoint: {destination}")
        return str(destination)
    print(f"Keepa service file endpoint failed for {filename}: {curl_result.stderr.strip()}")

    scp_result = subprocess.run(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            f"tubslamanna@anthonys-macbook-pro:{csv_path}",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if scp_result.returncode == 0 and destination.exists():
        print(f"Copied CSV from Mac over scp: {destination}")
        return str(destination)
    print(f"SCP CSV transfer failed for {csv_path}: {scp_result.stderr.strip()}")
    return csv_path


async def main() -> None:
    args = parse_args()
    print("=== LIVE E2E TEST START ===")
    print(f"Keepa URL: {KEEPA_URL}")

    if args.csv_path:
        print("=== KEEPA DOWNLOAD SKIPPED ===")
        print("Using CSV paths supplied with --csv-path.")
        download_output: dict[str, Any] = {"files": [{"output_path": path} for path in args.csv_path]}
    else:
        try:
            spec = get_agent("keepa-downloader")
            result = await execute_service_agent(spec, {"urls": [KEEPA_URL]})
        except Exception:
            print("=== KEEPA DOWNLOAD ERROR ===")
            traceback.print_exc()
            return

        download_output = result.output
        print("=== KEEPA DOWNLOAD RESULT ===")
        print_json(download_output)

    csv_paths = extract_csv_paths(download_output)
    print("=== EXTRACTED CSV PATHS ===")
    print_json(csv_paths)

    csv_paths = localize_csv_paths(csv_paths)
    print("=== LOCAL CSV PATHS ===")
    print_json(csv_paths)

    if not csv_paths:
        print("=== DATA RUNNER SKIPPED ===")
        print("No CSV file paths were returned by keepa-downloader.")
        return

    try:
        dr_spec = get_agent("data-runner")
        dr_result = await execute_llm_agent(
            dr_spec,
            {
                "brand": "live-test",
                "product_urls": [KEEPA_URL],
                "csv_paths": csv_paths,
                "download_result": download_output,
            },
        )
    except Exception:
        print("=== DATA RUNNER ERROR ===")
        traceback.print_exc()
        return

    print("=== DATA RUNNER RESULT ===")
    print_json(dr_result.output)
    print("=== DATA RUNNER USAGE ===")
    print_json(
        {
            "tokens_input": dr_result.tokens_input,
            "tokens_output": dr_result.tokens_output,
            "cost_usd": dr_result.cost_usd,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
