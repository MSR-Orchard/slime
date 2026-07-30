#!/usr/bin/env python3
"""
Delete leftover Orchard sandboxes that were not cleaned up properly
(e.g. after Ctrl+C).

Reads sandbox marker files written by ``azure_modal_docker.py`` from the
manifest directory (default: ``/tmp/.orchard_sandboxes/``),
deletes each sandbox via the sandbox API, and removes the marker file.

Usage:
    python scripts/cleanup_azure_sandboxes.py [--manifest-dir DIR]

Environment variables:
    ORCHARD_SANDBOX_MANIFEST_DIR  Override the default manifest directory.
    ORCHARD_SANDBOX_ENDPOINT      Fallback endpoint when a marker file has
                                  no endpoint recorded.
"""

import argparse
import asyncio
import json
import logging
import os

from orchard_env.client.sandbox_client import AsyncSandboxClient

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST_DIR = "/tmp/.orchard_sandboxes"


def _read_records(manifest_dir: str) -> list[dict[str, str]]:
    """Read all sandbox records from the manifest directory."""
    manifest_dir = os.path.abspath(manifest_dir)
    if not os.path.isdir(manifest_dir):
        return []
    records = []
    for name in os.listdir(manifest_dir):
        marker = os.path.join(manifest_dir, name)
        try:
            with open(marker) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            records.append({"sandbox_id": name, "endpoint": ""})
    return records


def _remove_marker(manifest_dir: str, sandbox_id: str) -> None:
    marker = os.path.join(manifest_dir, sandbox_id)
    try:
        os.remove(marker)
    except OSError:
        pass


async def cleanup(manifest_dir: str, fallback_endpoint: str | None = None) -> None:
    manifest_dir = os.path.abspath(manifest_dir)
    records = _read_records(manifest_dir)
    if not records:
        print(f"No sandbox markers found in {manifest_dir}")
        return

    print(f"Found {len(records)} leftover sandbox(es) to clean up:")
    for r in records:
        print(f"  - {r['sandbox_id']}  endpoint={r.get('endpoint', 'N/A')}")

    # Group by endpoint so we reuse one client per endpoint
    by_endpoint: dict[str, list[str]] = {}
    for r in records:
        ep = r.get("endpoint") or fallback_endpoint or ""
        if not ep:
            print(f"  WARNING: No endpoint for sandbox {r['sandbox_id']}, skipping. "
                  "Set ORCHARD_SANDBOX_ENDPOINT to provide a fallback.")
            continue
        by_endpoint.setdefault(ep, []).append(r["sandbox_id"])

    for endpoint, sandbox_ids in by_endpoint.items():
        client = AsyncSandboxClient(endpoint)
        try:
            for sid in sandbox_ids:
                try:
                    await client.delete_sandbox(sid)
                    print(f"  Deleted {sid}")
                except Exception as exc:
                    print(f"  Failed to delete {sid} (may already be gone): {exc}")
                _remove_marker(manifest_dir, sid)
        finally:
            try:
                if hasattr(client, "close"):
                    await client.close(cleanup=True)
                elif hasattr(client, "_session") and client._session and not client._session.closed:
                    await client._session.close()
            except Exception:
                pass

    print("Cleanup complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Delete leftover Orchard sandboxes from a previous run."
    )
    parser.add_argument(
        "--manifest-dir",
        default=os.environ.get("ORCHARD_SANDBOX_MANIFEST_DIR", _DEFAULT_MANIFEST_DIR),
        help="Directory containing sandbox marker files "
             "(default: /tmp/.orchard_sandboxes/ or ORCHARD_SANDBOX_MANIFEST_DIR)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fallback_endpoint = os.environ.get("ORCHARD_SANDBOX_ENDPOINT")
    asyncio.run(cleanup(args.manifest_dir, fallback_endpoint))


if __name__ == "__main__":
    main()
