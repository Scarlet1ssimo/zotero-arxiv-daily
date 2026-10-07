import json
from datetime import datetime, timezone

from zotero_arxiv_daily.daily_report import (
    build_daily_record,
    local_report_date,
    write_daily_record,
)
from tests.canned_responses import make_sample_paper


def test_local_report_date_uses_chicago_timezone():
    instant = datetime(2026, 10, 7, 4, 30, tzinfo=timezone.utc)
    assert local_report_date(instant) == "2026-10-06"


def test_daily_record_contains_candidates_and_ranked_recommendations(tmp_path):
    candidate = make_sample_paper(
        paper_id="2601.00001",
        categories=["cs.PL"],
        retrieval_channels=["semantic", "diversity"],
        channel_scores={"semantic": 0.7, "diversity": 0.4},
        assessment={"relevance": 8, "bucket": "direct"},
    )
    record = build_daily_record("2026-10-07", 120, [candidate], [candidate])
    path = write_daily_record(record, tmp_path / "data" / "daily")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["candidate_count"] == 1
    assert saved["recommendations"][0]["rank"] == 1
    assert saved["candidates"][0]["retrieval_channels"] == ["semantic", "diversity"]
    assert "full_text" not in saved["candidates"][0]


def test_same_day_record_is_overwritten(tmp_path):
    first = build_daily_record("2026-10-07", 1, [], [])
    second = build_daily_record("2026-10-07", 2, [], [])
    path = write_daily_record(first, tmp_path)
    write_daily_record(second, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["retrieved_count"] == 2
