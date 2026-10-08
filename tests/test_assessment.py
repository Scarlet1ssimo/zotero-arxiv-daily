import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.assessment import _parse_json_response, assess_candidates, build_slate
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


def test_assess_candidates_splits_large_candidate_list_into_batches():
    candidates = [
        Paper(
            source="arxiv",
            title=f"Paper {index}",
            authors=[],
            abstract="Abstract",
            url=f"https://arxiv.org/abs/2601.{index:05d}",
            paper_id=f"2601.{index:05d}",
        )
        for index in range(1, 4)
    ]
    batch_sizes = []

    def create(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        payload = json.loads(prompt.split("\n\n", 1)[1])
        papers = payload["candidate_papers"]
        batch_sizes.append(len(papers))
        assessments = [
            {
                "paper_id": paper["paper_id"],
                "relevance": 8,
                "novelty": 7,
                "elegance": 7,
                "transferability": 7,
                "confidence": 0.9,
                "bucket": "direct",
                "contribution": {"zh": "方法摘要。", "en": "Method summary."},
                "why_care": {"zh": "研究相关。", "en": "Relevant work."},
            }
            for paper in papers
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"assessments": assessments}))
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = assess_candidates(
        client,
        OmegaConf.create({"api_mode": "chat_completion", "generation_kwargs": {"model": "deepseek-flash"}}),
        candidates,
        "Compiler research",
        batch_size=2,
    )

    assert result == candidates
    assert batch_sizes == [2, 1]
    assert all(paper.assessment["relevance"] == 8 for paper in candidates)


def test_parse_json_response_explains_truncated_output():
    with pytest.raises(ValueError, match="incomplete JSON.*truncated"):
        _parse_json_response('{"assessments":[{"paper_id":"2601.00001')
