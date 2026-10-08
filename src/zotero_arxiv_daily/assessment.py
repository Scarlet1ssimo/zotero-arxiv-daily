"""Batch LLM assessment and quota-based recommendation slate construction."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from .protocol import Paper, _request_llm


BUCKET_ORDER = ("direct", "adjacent", "elegant", "wildcard", "contrarian")


def _paper_id(paper: Paper) -> str:
    return paper.paper_id or paper.url


def _parse_json_response(response: str) -> dict[str, Any]:
    text = response.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*|\s*" + re.escape(fence) + r"$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as original_error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                "DeepSeek returned incomplete JSON; the response may have been truncated"
            ) from original_error
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(
                "DeepSeek returned invalid or truncated JSON "
                f"(character {error.pos}: {error.msg})"
            ) from error
    if not isinstance(parsed, dict):
        raise ValueError("DeepSeek response must be a JSON object")
    return parsed


def _clamp_score(value: Any, name: str) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} score: {value!r}") from exc
    if not 1 <= score <= 10:
        raise ValueError(f"{name} score must be between 1 and 10")
    return score


def assess_candidates(
    client: Any,
    llm_config: Any,
    candidates: list[Paper],
    profile: str,
    recent_papers: list[Paper] | None = None,
    batch_size: int = 8,
) -> list[Paper]:
    if not candidates:
        return candidates
    if batch_size < 1:
        raise ValueError("Assessment batch_size must be at least 1")

    recent_context = [
        {"title": paper.title, "abstract": paper.abstract}
        for paper in (recent_papers or [])[:40]
    ]
    llm_params = dict(llm_config)
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))
    generation_kwargs.setdefault("response_format", {"type": "json_object"})

    system_prompt = """You are a careful research-paper curator for a researcher at the intersection of compilers, parallel computing, and computer architecture. Evaluate only the supplied title, abstract, categories, and research context. Do not invent paper claims. Be selective: a weak fit should receive relevance below 6. Assign exactly one bucket per paper: direct, adjacent, elegant, wildcard, or contrarian. Direct means close to active interests; adjacent means a useful neighboring method/problem; elegant means a particularly clean or promising idea; wildcard means plausible but outside the main concentration; contrarian means a paper whose formulation or evidence challenges common assumptions. Return one JSON object with an assessments array. For every supplied paper_id, return exactly one item with: paper_id, integer relevance, novelty, elegance, transferability (each 1-10), numeric confidence (0-1), bucket, and contribution and why_care objects each containing concise zh and en strings. Keep each description to one short sentence in each language. Output valid JSON only."""
    llm_params["generation_kwargs"] = generation_kwargs
    total_batches = (len(candidates) + batch_size - 1) // batch_size

    for batch_number, start in enumerate(range(0, len(candidates), batch_size), start=1):
        batch = candidates[start : start + batch_size]
        examples = [
            {
                "paper_id": _paper_id(paper),
                "title": paper.title,
                "authors": paper.authors[:8],
                "abstract": paper.abstract,
                "categories": paper.categories or [],
                "retrieval_channels": paper.retrieval_channels or [],
            }
            for paper in batch
        ]
        payload = {
            "research_profile": profile,
            "recent_zotero_papers": recent_context,
            "candidate_papers": examples,
        }
        user_prompt = "Assess these candidates comparatively. Keep descriptions concise.\n\n" + json.dumps(
            payload, ensure_ascii=False
        )
        logger.info(
            "Assessing candidate batch {}/{} ({} papers)",
            batch_number,
            total_batches,
            len(batch),
        )
        response = _request_llm(
            client,
            llm_params,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            parsed = _parse_json_response(response or "")
        except ValueError as exc:
            raise ValueError(
                f"DeepSeek assessment batch {batch_number}/{total_batches} failed: {exc}"
            ) from exc
        assessments = parsed.get("assessments")
        if not isinstance(assessments, list):
            raise ValueError(
                f"DeepSeek assessment batch {batch_number}/{total_batches} "
                "must contain an assessments array"
            )

        by_id = {_paper_id(paper): paper for paper in batch}
        seen: set[str] = set()
        for item in assessments:
            if not isinstance(item, dict):
                continue
            identity = str(item.get("paper_id", ""))
            paper = by_id.get(identity)
            if paper is None or identity in seen:
                continue
            bucket = str(item.get("bucket", "")).casefold()
            if bucket not in BUCKET_ORDER:
                logger.warning("Ignoring assessment with invalid bucket for {}", identity)
                continue
            contribution = item.get("contribution") if isinstance(item.get("contribution"), dict) else {}
            why_care = item.get("why_care") if isinstance(item.get("why_care"), dict) else {}
            try:
                normalized = {
                    "relevance": _clamp_score(item.get("relevance"), "relevance"),
                    "novelty": _clamp_score(item.get("novelty"), "novelty"),
                    "elegance": _clamp_score(item.get("elegance"), "elegance"),
                    "transferability": _clamp_score(item.get("transferability"), "transferability"),
                    "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                    "bucket": bucket,
                    "contribution": {
                        "zh": str(contribution.get("zh", "")),
                        "en": str(contribution.get("en", "")),
                    },
                    "why_care": {
                        "zh": str(why_care.get("zh", "")),
                        "en": str(why_care.get("en", "")),
                    },
                }
            except (ValueError, TypeError) as exc:
                logger.warning("Ignoring invalid assessment for {}: {}", identity, exc)
                continue
            paper.assessment = normalized
            paper.score = float(normalized["relevance"])
            paper.tldr = normalized["contribution"]["zh"] or paper.abstract
            seen.add(identity)

        missing = sorted(set(by_id) - seen)
        if missing:
            raise ValueError(
                f"DeepSeek omitted or returned invalid assessments for "
                f"{len(missing)} candidates in batch {batch_number}/{total_batches}"
            )
    return candidates


def build_slate(candidates: list[Paper], config: Any) -> list[Paper]:
    """Select each configured bucket independently, respecting relevance floor."""
    slate_config = config.get("slate", {})
    quota_config = slate_config.get("quotas", {})
    minimum_relevance = int(slate_config.get("minimum_relevance", 6))
    selected: list[Paper] = []
    for bucket in BUCKET_ORDER:
        quota = int(quota_config.get(bucket, 0))
        eligible = [
            paper
            for paper in candidates
            if paper.assessment
            and paper.assessment.get("bucket") == bucket
            and int(paper.assessment.get("relevance", 0)) >= minimum_relevance
            and paper not in selected
        ]
        eligible.sort(
            key=lambda paper: (
                -int(paper.assessment.get("relevance", 0)),
                -int(paper.assessment.get("novelty", 0)),
                _paper_id(paper),
            )
        )
        selected.extend(eligible[:quota])
    return selected[: int(slate_config.get("max_papers", 10))]
