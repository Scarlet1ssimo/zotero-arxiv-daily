"""Category gating and multi-channel arXiv candidate generation."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .protocol import CorpusPaper, Paper


CHANNELS = ("semantic", "lexical", "diversity", "exploration")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


@dataclass
class CandidatePool:
    papers: list[Paper]
    channel_pools: dict[str, list[Paper]]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return list(value)


def _paper_identity(paper: Paper) -> str:
    if paper.paper_id:
        return re.sub(r"v\d+$", "", paper.paper_id)
    return paper.url or paper.title.casefold().strip()


def _topic_text(paper: Paper) -> str:
    return f"{paper.title or ''}\n{paper.abstract or ''}"


def _term_pattern(term: str) -> re.Pattern:
    escaped = re.escape(term.strip().casefold())
    escaped = escaped.replace(r"\ ", r"[\s-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def matching_keyword_groups(paper: Paper, keyword_groups: dict[str, Any]) -> list[str]:
    text = _topic_text(paper)
    matched = []
    for group, terms in keyword_groups.items():
        if any(_term_pattern(str(term)).search(text) for term in _as_list(terms)):
            matched.append(str(group))
    return matched


def filter_by_category(papers: list[Paper], config: Any) -> list[Paper]:
    """Keep core-category papers, plus keyword-gated peripheral papers."""
    core = set(_as_list(config.get("core_categories", [])))
    peripheral = set(_as_list(config.get("peripheral_categories", [])))
    keyword_groups = config.get("keyword_groups", {}) or {}
    accepted = []
    for paper in papers:
        categories = set(paper.categories or [])
        if categories & core:
            accepted.append(paper)
            continue
        if categories & peripheral and matching_keyword_groups(paper, keyword_groups):
            accepted.append(paper)
    return accepted


def _tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Small dependency-free BM25 implementation for one research-profile query."""
    if not documents:
        return np.array([], dtype=float)
    query_tokens = _tokenize(query)
    tokenized = [_tokenize(document) for document in documents]
    count = len(tokenized)
    average_length = sum(map(len, tokenized)) / max(count, 1)
    document_frequency: dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for token in set(tokens):
            document_frequency[token] += 1

    scores = np.zeros(count, dtype=float)
    for index, tokens in enumerate(tokenized):
        term_frequency: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_frequency[token] += 1
        length = len(tokens)
        for term in dict.fromkeys(query_tokens):
            frequency = term_frequency.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = np.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / max(average_length, 1))
            scores[index] += inverse_frequency * frequency * (k1 + 1) / denominator
    return scores


def _normalized(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores
    low, high = float(np.min(scores)), float(np.max(scores))
    if high - low < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / (high - low)


def _rank_indices(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


class CandidateGenerator:
    def __init__(self, config: Any, encoder: Any = None):
        self.config = config
        self.encoder = encoder

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                self.config.reranker.local.model,
                trust_remote_code=True,
            )
        return self.encoder

    def _semantic_scores(
        self, papers: list[Paper], corpus: list[CorpusPaper], profile: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if not papers:
            return np.array([]), np.empty((0, 0))
        documents = [_topic_text(paper) for paper in papers]
        context = [f"{paper.title}\n{paper.abstract}" for paper in corpus if paper.title or paper.abstract]
        if not context:
            context = [profile] if profile.strip() else [""]

        encode_kwargs = self.config.reranker.local.get("encode_kwargs", {}) or {}
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(encode_kwargs):
                encode_kwargs = OmegaConf.to_container(encode_kwargs, resolve=True)
        except ImportError:
            encode_kwargs = dict(encode_kwargs)
        encode_kwargs = dict(encode_kwargs)
        encode_kwargs.pop("show_progress_bar", None)
        encoder = self._get_encoder()
        paper_vectors = encoder.encode(documents, show_progress_bar=True, **encode_kwargs)
        context_vectors = encoder.encode(context, show_progress_bar=True, **encode_kwargs)
        paper_vectors = np.asarray(paper_vectors)
        context_vectors = np.asarray(context_vectors)
        context_similarity = cosine_similarity(paper_vectors, context_vectors)
        top_n = min(3, context_similarity.shape[1])
        scores = np.sort(context_similarity, axis=1)[:, -top_n:].mean(axis=1)
        pairwise_similarity = cosine_similarity(paper_vectors)
        return scores, pairwise_similarity

    def generate(
        self,
        papers: list[Paper],
        corpus: list[CorpusPaper],
        run_date: str,
    ) -> CandidatePool:
        retrieval_config = self.config.get("research", {})
        eligible = filter_by_category(papers, retrieval_config)
        if not eligible:
            return CandidatePool([], {channel: [] for channel in CHANNELS})

        recent_count = int(retrieval_config.get("recent_zotero_count", 40))
        recent_corpus = sorted(corpus, key=lambda item: item.added_date, reverse=True)[:recent_count]
        profile = str(retrieval_config.get("profile", ""))
        semantic_scores, pairwise = self._semantic_scores(eligible, recent_corpus, profile)

        lexical_query_parts = [profile]
        lexical_query_parts.extend(item.title for item in recent_corpus)
        for terms in (retrieval_config.get("keyword_groups", {}) or {}).values():
            lexical_query_parts.extend(str(term) for term in terms)
        lexical_scores = bm25_scores(
            " ".join(part for part in lexical_query_parts if part),
            [_topic_text(paper) for paper in eligible],
        )

        budgets = retrieval_config.get("candidate_budgets", {})
        budgets = {channel: int(budgets.get(channel, 0)) for channel in CHANNELS}
        channel_pools: dict[str, list[Paper]] = {}
        channel_indices: dict[str, list[int]] = {
            "semantic": _rank_indices(semantic_scores)[: budgets["semantic"]],
            "lexical": _rank_indices(lexical_scores)[: budgets["lexical"]],
            "diversity": [],
            "exploration": [],
        }

        # MMR is an explicit retrieval channel: select relevant candidates while
        # penalizing similarity to candidates already selected in this channel.
        mmr_budget = budgets["diversity"]
        relevance = _normalized(semantic_scores)
        selected: list[int] = []
        remaining = set(range(len(eligible)))
        mmr_lambda = float(retrieval_config.get("mmr_lambda", 0.72))
        mmr_scores: dict[int, float] = {}
        while remaining and len(selected) < mmr_budget:
            def mmr_value(index: int) -> tuple[float, int]:
                redundancy = max((float(pairwise[index, picked]) for picked in selected), default=0.0)
                return mmr_lambda * float(relevance[index]) - (1 - mmr_lambda) * redundancy, -index

            chosen = max(remaining, key=mmr_value)
            selected.append(chosen)
            mmr_scores[chosen] = mmr_value(chosen)[0]
            remaining.remove(chosen)
        channel_indices["diversity"] = selected

        # Exploration is reproducible for a local run date and samples across
        # categories instead of drawing uniformly from an unstructured pool.
        exploration_budget = budgets["exploration"]
        semantic_order = _rank_indices(semantic_scores)
        obvious_count = max(1, int(len(semantic_order) * 0.30))
        core_categories = set(_as_list(retrieval_config.get("core_categories", [])))
        plausible = [
            i
            for i in semantic_order[obvious_count:]
            if set(eligible[i].categories or []) & core_categories
            or matching_keyword_groups(
                eligible[i], retrieval_config.get("keyword_groups", {}) or {}
            )
        ]
        if not plausible:
            plausible = semantic_order[obvious_count:] or semantic_order
        explored_channels = set(
            index
            for channel in ("semantic", "lexical", "diversity")
            for index in channel_indices[channel]
        )
        unexplored = [index for index in plausible if index not in explored_channels]
        if unexplored:
            plausible = unexplored
        groups: dict[str, list[int]] = defaultdict(list)
        for index in plausible:
            category_key = next(iter(sorted(eligible[index].categories or [])), "uncategorized")
            groups[category_key].append(index)
        seed = int.from_bytes(hashlib.sha256(run_date.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)
        for indices in groups.values():
            rng.shuffle(indices)
        ordered_groups = sorted(groups)
        exploration = []
        while ordered_groups and len(exploration) < exploration_budget:
            next_round = []
            for group in ordered_groups:
                if groups[group]:
                    exploration.append(groups[group].pop())
                    if groups[group]:
                        next_round.append(group)
                    if len(exploration) >= exploration_budget:
                        break
            ordered_groups = next_round
        channel_indices["exploration"] = exploration

        identity_to_channels: dict[str, list[str]] = defaultdict(list)
        identity_to_scores: dict[str, dict[str, float]] = defaultdict(dict)
        identity_to_paper: dict[str, Paper] = {}
        for channel in CHANNELS:
            selected_papers = []
            indices = channel_indices[channel]
            channel_scores = {
                "semantic": semantic_scores,
                "lexical": lexical_scores,
                "diversity": np.asarray(
                    [mmr_scores.get(index, 0.0) for index in range(len(eligible))]
                ),
                "exploration": 1.0 - relevance,
            }[channel]
            for index in indices:
                paper = eligible[index]
                identity = _paper_identity(paper)
                if channel not in identity_to_channels[identity]:
                    identity_to_channels[identity].append(channel)
                identity_to_scores[identity][channel] = float(channel_scores[index])
                identity_to_paper[identity] = paper
                selected_papers.append(paper)
            channel_pools[channel] = selected_papers

        ordered_identities = []
        for channel in CHANNELS:
            for index in channel_indices[channel]:
                identity = _paper_identity(eligible[index])
                if identity not in ordered_identities:
                    ordered_identities.append(identity)

        candidates = []
        for identity in ordered_identities:
            paper = identity_to_paper[identity]
            paper.retrieval_channels = identity_to_channels[identity]
            paper.channel_scores = identity_to_scores[identity]
            candidates.append(paper)
        return CandidatePool(candidates, channel_pools)
