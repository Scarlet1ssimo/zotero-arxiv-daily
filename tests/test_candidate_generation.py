from datetime import datetime

import numpy as np

from zotero_arxiv_daily.candidate_generation import (
    CandidateGenerator,
    filter_by_category,
    matching_keyword_groups,
)
from zotero_arxiv_daily.protocol import CorpusPaper, Paper


def paper(title, abstract, categories, paper_id):
    return Paper(
        source="arxiv",
        title=title,
        authors=["A. Author"],
        abstract=abstract,
        url=f"https://arxiv.org/abs/{paper_id}",
        paper_id=paper_id,
        categories=categories,
    )


def research_config():
    return {
        "core_categories": ["cs.PL", "cs.AR"],
        "peripheral_categories": ["cs.AI", "eess.SP"],
        "keyword_groups": {
            "compiler_ir": ["compiler", "intermediate representation"],
            "pim": ["PIM", "processing-in-memory"],
        },
        "recent_zotero_count": 40,
        "candidate_budgets": {
            "semantic": 2,
            "lexical": 2,
            "diversity": 1,
            "exploration": 1,
        },
        "mmr_lambda": 0.7,
    }


def test_core_categories_are_kept_without_keyword_match():
    papers = [
        paper("An unrelated sounding paper", "No configured keyword.", ["cs.PL"], "1"),
        paper("Unrelated AI work", "No configured keyword.", ["cs.AI"], "2"),
    ]
    assert [item.paper_id for item in filter_by_category(papers, research_config())] == ["1"]


def test_peripheral_category_requires_keyword_and_checks_all_categories():
    candidate = paper("Memory placement", "We compile kernels for a PIM device.", ["cs.AI", "cs.CV"], "1")
    no_match = paper("Image segmentation", "A model for images.", ["cs.AI"], "2")
    assert filter_by_category([candidate, no_match], research_config()) == [candidate]


def test_keyword_matching_uses_word_boundaries_and_hyphenated_phrases():
    assert matching_keyword_groups(
        paper("PIM acceleration", "processing-in-memory system", ["cs.AI"], "1"),
        research_config()["keyword_groups"],
    ) == ["pim"]
    assert not matching_keyword_groups(
        paper("Optimization", "A simple approach to image prediction.", ["cs.AI"], "2"),
        {"gpu": ["GPU"], "pim": ["PIM"]},
    )


class TinyEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vectors.append(
                [
                    1 + lowered.count("compiler"),
                    1 + lowered.count("gpu"),
                    1 + lowered.count("memory"),
                    1 + len(text) % 11,
                ]
            )
        return np.asarray(vectors, dtype=float)


def _generator():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "research": research_config(),
            "reranker": {
                "local": {
                    "model": "unused",
                    "encode_kwargs": {},
                }
            },
        }
    )
    return CandidateGenerator(config, encoder=TinyEncoder())


def test_candidate_channels_keep_overlap_provenance_and_exploration_is_repeatable():
    candidates = [
        paper("Compiler GPU kernel", "Compiler generates GPU code.", ["cs.PL"], "1"),
        paper("Compiler IR rewrite", "Compiler uses an intermediate representation.", ["cs.PL"], "2"),
        paper("PIM scheduling", "Mapping a compiler schedule to processing-in-memory.", ["cs.AR"], "3"),
        paper("Compiler memory layout", "Compiler optimizes a memory layout.", ["cs.PL"], "4"),
        paper("Compiler on a PIM device", "Compiler maps work to PIM.", ["cs.AI"], "5"),
    ]
    corpus = [
        CorpusPaper(
            title="Compiler IR",
            abstract="Compiler intermediate representation.",
            added_date=datetime(2026, 10, 1),
            paths=[],
        )
    ]
    pool_one = _generator().generate(candidates, corpus, "2026-10-07")
    pools = {channel: {item.paper_id for item in items} for channel, items in pool_one.channel_pools.items()}
    assert pools["semantic"]
    assert pools["lexical"]
    assert pools["diversity"]
    assert pools["exploration"]
    overlap = next(item for item in pool_one.papers if len(item.retrieval_channels or []) > 1)
    assert len(overlap.retrieval_channels) > 1

    candidates_again = [
        paper("Compiler GPU kernel", "Compiler generates GPU code.", ["cs.PL"], "1"),
        paper("Compiler IR rewrite", "Compiler uses an intermediate representation.", ["cs.PL"], "2"),
        paper("PIM scheduling", "Mapping a compiler schedule to processing-in-memory.", ["cs.AR"], "3"),
        paper("Compiler memory layout", "Compiler optimizes a memory layout.", ["cs.PL"], "4"),
        paper("Compiler on a PIM device", "Compiler maps work to PIM.", ["cs.AI"], "5"),
    ]
    pool_two = _generator().generate(candidates_again, corpus, "2026-10-07")
    assert [item.paper_id for item in pool_one.channel_pools["exploration"]] == [
        item.paper_id for item in pool_two.channel_pools["exploration"]
    ]
