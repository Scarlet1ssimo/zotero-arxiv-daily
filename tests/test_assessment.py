import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.assessment import assess_candidates, build_slate
from zotero_arxiv_daily.protocol import Paper


def make_paper(identifier, bucket, relevance):
    return Paper(
        source="arxiv",
        title=f"Paper {identifier}",
        authors=["A. Author"],
        abstract="An abstract.",
        url=f"https://arxiv.org/abs/{identifier}",
        paper_id=identifier,
        categories=["cs.PL"],
        assessment={
            "bucket": bucket,
            "relevance": relevance,
            "novelty": 8,
            "elegance": 8,
            "transferability": 7,
        },
    )


def test_build_slate_applies_quality_floor_and_bucket_quotas():
    papers = [
        make_paper("1", "direct", 9),
        make_paper("2", "direct", 6),
        make_paper("3", "direct", 5),
        make_paper("4", "adjacent", 8),
        make_paper("5", "wildcard", 7),
        make_paper("6", "contrarian", 9),
    ]
    config = OmegaConf.create(
        {
            "slate": {
                "max_papers": 10,
                "minimum_relevance": 6,
                "quotas": {
                    "direct": 1,
                    "adjacent": 2,
                    "elegant": 2,
                    "wildcard": 1,
                    "contrarian": 1,
                },
            }
        }
    )
    slate = build_slate(papers, config)
    assert [paper.paper_id for paper in slate] == ["1", "4", "5", "6"]
    assert all(paper.assessment["relevance"] >= 6 for paper in slate)


def test_assess_candidates_sends_openai_compatible_json_request():
    response = {
        "assessments": [
            {
                "paper_id": "2601.00001",
                "relevance": 8,
                "novelty": 7,
                "elegance": 8,
                "transferability": 6,
                "confidence": 0.9,
                "bucket": "direct",
                "contribution": {"zh": "提出编译方法。", "en": "Presents a compiler method."},
                "why_care": {"zh": "贴近兴趣。", "en": "Matches current interests."},
            }
        ]
    }
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)), finish_reason="stop")]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    candidate = Paper(
        source="arxiv",
        title="Compiler transformations",
        authors=["A. Author"],
        abstract="We optimize a compiler.",
        url="https://arxiv.org/abs/2601.00001",
        paper_id="2601.00001",
        categories=["cs.PL"],
        retrieval_channels=["semantic"],
    )
    llm_config = OmegaConf.create(
        {"api_mode": "chat_completion", "generation_kwargs": {"model": "deepseek-flash"}}
    )
    result = assess_candidates(client, llm_config, [candidate], "Compiler optimization")
    assert result == [candidate]
    assert captured["model"] == "deepseek-flash"
    assert captured["response_format"] == {"type": "json_object"}
    assert candidate.score == 8
    assert candidate.assessment["contribution"]["zh"] == "提出编译方法。"


def test_assess_candidates_fails_when_model_omits_candidates():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"assessments":[]}'))]
                )
            )
        )
    )
    candidate = Paper(
        source="arxiv",
        title="Title",
        authors=[],
        abstract="Abstract",
        url="https://arxiv.org/abs/2601.00002",
        paper_id="2601.00002",
    )
    with pytest.raises(ValueError, match="omitted or returned invalid assessments"):
        assess_candidates(
            client,
            OmegaConf.create({"api_mode": "chat_completion", "generation_kwargs": {"model": "deepseek-flash"}}),
            [candidate],
            "Compiler",
        )
