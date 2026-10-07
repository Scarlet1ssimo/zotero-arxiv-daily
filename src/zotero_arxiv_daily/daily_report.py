"""Date-keyed JSON snapshots for the daily research-paper radar."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zoneinfo import ZoneInfo

from .protocol import Paper


def local_report_date(
    now: datetime | None = None, timezone_name: str = "America/Chicago"
) -> str:
    timezone = ZoneInfo(timezone_name)
    current = now or datetime.now(timezone)
    return current.astimezone(timezone).date().isoformat()


def paper_to_record(paper: Paper, *, rank: int | None = None) -> dict[str, Any]:
    record = {
        "paper_id": paper.paper_id or paper.url,
        "source": paper.source,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "categories": paper.categories or [],
        "published_at": paper.published_date.isoformat() if paper.published_date else None,
        "retrieval_channels": paper.retrieval_channels or [],
        "channel_scores": paper.channel_scores or {},
        "assessment": paper.assessment,
    }
    if rank is not None:
        record["rank"] = rank
    return record


def build_daily_record(
    report_date: str,
    retrieved_count: int,
    candidates: list[Paper],
    recommendations: list[Paper],
    *,
    status: str = "complete",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "date": report_date,
        "status": status,
        "retrieved_count": retrieved_count,
        "candidate_count": len(candidates),
        "recommendation_count": len(recommendations),
        "candidates": [paper_to_record(paper) for paper in candidates],
        "recommendations": [
            paper_to_record(paper, rank=index)
            for index, paper in enumerate(recommendations, start=1)
        ],
        "error": error,
    }


def write_daily_record(record: dict[str, Any], data_dir: str | Path = "data/daily") -> Path:
    report_date = str(record["date"])
    path = Path(data_dir) / f"{report_date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    return path
