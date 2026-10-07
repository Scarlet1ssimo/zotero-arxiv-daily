from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .candidate_generation import CandidateGenerator
from .assessment import assess_candidates, build_slate
from .daily_report import (
    build_daily_record,
    local_report_date,
    write_daily_record,
)
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.candidate_generator = CandidateGenerator(config)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [
            item
            for item in corpus
            if item.get('data', {}).get('title', '').strip()
            and item.get('data', {}).get('abstractNote', '').strip()
        ]
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        report_date = local_report_date(
            timezone_name=str(self.config.get("research", {}).get("timezone", "America/Chicago"))
        )
        retrieved_count = 0
        candidates = []
        recommendations = []
        record_path = None
        try:
            corpus = self.fetch_zotero_corpus()
            corpus = self.filter_corpus(corpus)
        except Exception as exc:
            write_daily_record(
                build_daily_record(
                    report_date,
                    retrieved_count,
                    candidates,
                    recommendations,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
                self.config.get("research", {}).get("daily_dir", "data/daily"),
            )
            raise
        if len(corpus) == 0:
            error = "No Zotero papers found. Check Zotero credentials and collection filters."
            logger.error(error)
            write_daily_record(
                build_daily_record(
                    report_date,
                    retrieved_count,
                    candidates,
                    recommendations,
                    status="failed",
                    error=error,
                ),
                self.config.get("research", {}).get("daily_dir", "data/daily"),
            )
            raise ValueError(error)
        all_papers = []
        try:
            for source, retriever in self.retrievers.items():
                logger.info(f"Retrieving {source} papers...")
                papers = retriever.retrieve_papers()
                if len(papers) == 0:
                    logger.info(f"No {source} papers found")
                    continue
                logger.info(f"Retrieved {len(papers)} {source} papers")
                all_papers.extend(papers)
            retrieved_count = len(all_papers)
        except Exception as exc:
            write_daily_record(
                build_daily_record(
                    report_date,
                    retrieved_count,
                    candidates,
                    recommendations,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                ),
                self.config.get("research", {}).get("daily_dir", "data/daily"),
            )
            raise
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        try:
            logger.info("Applying category and keyword gates, then generating candidates...")
            pool = self.candidate_generator.generate(all_papers, corpus, report_date)
            candidates = pool.papers
            logger.info(f"Assessing {len(candidates)} unique candidates with DeepSeek...")
            research_config = self.config.get("research", {})
            assess_candidates(
                self.openai_client,
                self.config.llm,
                candidates,
                str(research_config.get("profile", "")),
                recent_papers=sorted(corpus, key=lambda item: item.added_date, reverse=True)[
                    : int(research_config.get("recent_zotero_count", 40))
                ],
            )
            recommendations = build_slate(candidates, self.config.get("research", {}))
        except Exception as exc:
            failed_record = build_daily_record(
                report_date,
                retrieved_count,
                candidates,
                recommendations,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            record_path = write_daily_record(
                failed_record,
                self.config.get("research", {}).get("daily_dir", "data/daily"),
            )
            logger.exception(f"Daily assessment failed; saved partial record to {record_path}")
            raise

        record = build_daily_record(report_date, retrieved_count, candidates, recommendations)
        record["delivery_status"] = "pending"
        data_dir = self.config.get("research", {}).get("daily_dir", "data/daily")
        record_path = write_daily_record(record, data_dir)
        logger.info(f"Saved daily paper record to {record_path}")

        should_send = bool(recommendations) or bool(self.config.executor.send_empty)
        if not should_send:
            record["delivery_status"] = "skipped_empty"
            write_daily_record(record, data_dir)
            logger.info("No qualifying recommendations; no email will be sent.")
            return

        try:
            logger.info(f"Sending {len(recommendations)} recommendations by email...")
            send_email(self.config, render_email(recommendations), report_date=report_date)
            record["delivery_status"] = "sent"
            write_daily_record(record, data_dir)
            logger.info("Email sent successfully")
        except Exception as exc:
            record["delivery_status"] = "failed"
            record["delivery_error"] = f"{type(exc).__name__}: {exc}"
            write_daily_record(record, data_dir)
            raise
