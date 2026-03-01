from __future__ import annotations

import argparse
import json
import re
import time
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_QUERIES = [
    "maintenance reel",
    "aircraft maintenance",
    "equipment inspection",
    "cleaning TESDA",
]
DEFAULT_INCLUDE_KEYWORDS = [
    "maintenance",
    "clean",
    "cleaning",
    "inspection",
    "inspect",
    "filter",
    "lubric",
    "sanitize",
    "wash",
    "disinfect",
    "hygiene",
    "reel",
]
DEFAULT_EXCLUDE_KEYWORDS = [
    "abyss",
    "poisson",
    "station seamon",
    "space station",
    "ifremer",
    "auction",
    "super bowl",
    "interview footage",
]
ALLOWED_LICENSE_SNIPPETS = (
    "public domain",
    "cc by",
    "creative commons attribution",
    "cc0",
)


def _strip_html(value: str | None) -> str:
    if not value:
        return ""
    no_tags = re.sub(r"<[^>]+>", "", value)
    return unescape(no_tags).strip()


def _sanitize_filename(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized[:180] or "video"


def _load_custom_queries(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_QUERIES)
    merged: list[str] = []
    for raw in values:
        if not raw:
            continue
        parts = [part.strip() for part in raw.split(",")]
        merged.extend([part for part in parts if part])
    return merged or list(DEFAULT_QUERIES)


def _load_keyword_list(values: list[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return list(defaults)
    merged: list[str] = []
    for raw in values:
        if not raw:
            continue
        parts = [part.strip().lower() for part in raw.split(",")]
        merged.extend([part for part in parts if part])
    return merged or list(defaults)


def search_videos(
    client: httpx.Client,
    query_text: str,
    per_query_limit: int,
    max_size_mb: float,
    allow_unfiltered_license: bool,
    include_keywords: list[str],
    exclude_keywords: list[str],
) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:video {query_text}",
        "gsrnamespace": "6",
        "gsrlimit": str(min(max(1, per_query_limit), 50)),
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|mime|size|extmetadata",
    }
    response = client.get(COMMONS_API, params=params)
    response.raise_for_status()
    payload = response.json()
    pages = (payload.get("query") or {}).get("pages") or {}

    items: list[dict[str, Any]] = []
    for page in pages.values():
        image_info = ((page.get("imageinfo") or [{}])[0]) or {}
        mime = str(image_info.get("mime") or "")
        if not mime.startswith("video/"):
            continue

        size_bytes = int(image_info.get("size") or 0)
        if max_size_mb > 0 and size_bytes > int(max_size_mb * 1024 * 1024):
            continue

        ext = image_info.get("extmetadata") or {}
        license_short = _strip_html((ext.get("LicenseShortName") or {}).get("value"))
        license_url = _strip_html((ext.get("LicenseUrl") or {}).get("value"))
        artist = _strip_html((ext.get("Artist") or {}).get("value"))
        credit = _strip_html((ext.get("Credit") or {}).get("value"))
        description = _strip_html((ext.get("ImageDescription") or {}).get("value"))
        usage_terms = _strip_html((ext.get("UsageTerms") or {}).get("value"))

        license_text = f"{license_short} {usage_terms}".strip().lower()
        if not allow_unfiltered_license and not any(
            snippet in license_text for snippet in ALLOWED_LICENSE_SNIPPETS
        ):
            continue

        title = str(page.get("title") or "")
        if not title.startswith("File:"):
            continue
        file_name = title.split(":", 1)[1]
        download_url = str(image_info.get("url") or "")
        if not download_url:
            continue

        searchable_text = " ".join(
            [
                title.lower(),
                file_name.lower(),
                description.lower(),
                credit.lower(),
            ]
        )
        if include_keywords and not any(keyword in searchable_text for keyword in include_keywords):
            continue
        if exclude_keywords and any(keyword in searchable_text for keyword in exclude_keywords):
            continue

        items.append(
            {
                "query": query_text,
                "title": title,
                "file_name": file_name,
                "download_url": download_url,
                "description_url": f"https://commons.wikimedia.org/wiki/{quote(title, safe=':()_')}",
                "mime": mime,
                "size_bytes": size_bytes,
                "license_short": license_short,
                "license_url": license_url,
                "usage_terms": usage_terms,
                "artist": artist,
                "credit": credit,
                "description": description,
            }
        )

    return items


def _pick_extension(file_name: str, mime: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix:
        return suffix
    if mime == "video/webm":
        return ".webm"
    if mime == "video/mp4":
        return ".mp4"
    return ".video"


def download_file(client: httpx.Client, url: str, target: Path) -> int:
    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with target.open("wb") as fh:
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                fh.write(chunk)
                written += len(chunk)
        return written


def resolve_target_path(
    *,
    idx: int,
    output_dir: Path,
    layout: str,
    gold_count: int,
    file_name: str,
    mime: str,
) -> tuple[Path, str]:
    extension = _pick_extension(file_name, mime)
    base_name = _sanitize_filename(Path(file_name).stem)
    if layout == "split":
        is_gold = idx <= max(0, int(gold_count))
        subset = "gold" if is_gold else "trainee"
        local_name = f"{subset}_{idx:03d}_{base_name}{extension}"
        return output_dir / subset / local_name, subset
    local_name = f"{idx:03d}_{base_name}{extension}"
    return output_dir / local_name, "candidate"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch publicly licensed SOP-like videos from Wikimedia Commons for PoC seeding."
    )
    parser.add_argument("--output-dir", default="poc_videos/candidates", help="Destination folder")
    parser.add_argument(
        "--layout",
        choices=["candidates", "split"],
        default="candidates",
        help="candidates: save in one folder / split: save as gold+trainee subfolders",
    )
    parser.add_argument("--gold-count", type=int, default=20, help="Used only when --layout split")
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Search query term. Repeatable. Supports comma-separated input.",
    )
    parser.add_argument("--per-query-limit", type=int, default=25, help="Candidates to pull per query")
    parser.add_argument("--max-files", type=int, default=40, help="Maximum files to download")
    parser.add_argument("--max-size-mb", type=float, default=120.0, help="Skip larger files")
    parser.add_argument("--skip-license-filter", action="store_true", help="Do not filter by license text")
    parser.add_argument(
        "--include-keyword",
        action="append",
        default=None,
        help="Keep results containing these keywords (repeatable / comma-separated)",
    )
    parser.add_argument(
        "--exclude-keyword",
        action="append",
        default=None,
        help="Drop results containing these keywords (repeatable / comma-separated)",
    )
    parser.add_argument("--sleep-ms", type=int, default=200, help="Wait between downloads")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default="data/commons_seed_manifest.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_custom_queries(args.query)
    include_keywords = _load_keyword_list(args.include_keyword, DEFAULT_INCLUDE_KEYWORDS)
    exclude_keywords = _load_keyword_list(args.exclude_keyword, DEFAULT_EXCLUDE_KEYWORDS)
    user_agent = "sopilot-poc/0.1 (+https://github.com/)"
    timeout = httpx.Timeout(120.0, connect=15.0)

    all_candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    with httpx.Client(timeout=timeout, headers={"User-Agent": user_agent}) as client:
        for query in queries:
            try:
                items = search_videos(
                    client=client,
                    query_text=query,
                    per_query_limit=max(1, args.per_query_limit),
                    max_size_mb=max(0.0, float(args.max_size_mb)),
                    allow_unfiltered_license=args.skip_license_filter,
                    include_keywords=include_keywords,
                    exclude_keywords=exclude_keywords,
                )
            except Exception as exc:
                print(f"[search-fail] query={query!r} error={exc}")
                continue

            print(f"[search] query={query!r} candidates={len(items)}")
            for item in items:
                title = item["title"]
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                all_candidates.append(item)

    if not all_candidates:
        raise SystemExit("no candidate videos found")

    selected = all_candidates[: max(1, int(args.max_files))]
    print(f"selected={len(selected)} output_dir={output_dir}")

    manifest_rows: list[dict[str, Any]] = []

    if args.dry_run:
        for idx, item in enumerate(selected, start=1):
            target, subset = resolve_target_path(
                idx=idx,
                output_dir=output_dir,
                layout=args.layout,
                gold_count=args.gold_count,
                file_name=item["file_name"],
                mime=item["mime"],
            )
            print(
                f"[dry {idx:03d}] {item['file_name']} | {item['license_short'] or 'unknown'} | {item['download_url']}"
            )
            manifest_rows.append(
                {
                    **item,
                    "saved_as": str(target),
                    "subset": subset,
                    "downloaded_bytes": 0,
                    "status": "dry_run",
                }
            )
    else:
        with httpx.Client(timeout=timeout, headers={"User-Agent": user_agent}) as client:
            for idx, item in enumerate(selected, start=1):
                target, subset = resolve_target_path(
                    idx=idx,
                    output_dir=output_dir,
                    layout=args.layout,
                    gold_count=args.gold_count,
                    file_name=item["file_name"],
                    mime=item["mime"],
                )

                print(f"[download {idx:03d}/{len(selected)}] {item['file_name']}")
                row = dict(item)
                row["saved_as"] = str(target)
                row["subset"] = subset
                row["downloaded_bytes"] = 0
                row["status"] = "failed"
                try:
                    written = download_file(client, item["download_url"], target)
                    row["downloaded_bytes"] = written
                    row["status"] = "downloaded"
                    print(f"  -> ok bytes={written}")
                except Exception as exc:
                    print(f"  -> fail error={exc}")
                manifest_rows.append(row)
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

    manifest = {
        "source": "wikimedia_commons",
        "queries": queries,
        "include_keywords": include_keywords,
        "exclude_keywords": exclude_keywords,
        "selected_count": len(selected),
        "downloaded_count": sum(1 for row in manifest_rows if row["status"] == "downloaded"),
        "items": manifest_rows,
    }
    manifest_path = Path(args.manifest).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
