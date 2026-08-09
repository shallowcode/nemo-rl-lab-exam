from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[._+\-/][a-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_MIXED_IDENTIFIER = re.compile(
    r"(?i)(?<![a-z0-9])[a-z0-9][a-z0-9._-]{2,}(?![a-z0-9])"
)
_UPPER_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,9}(?![A-Za-z0-9])")
_QUESTION = re.compile(
    r"题目\s*[：:]\s*(.*?)(?:\n\s*\n|<\|im_end\|>|\Z)", re.DOTALL
)
_GENERIC_SEARCH_QUERIES = frozenset(
    {
        "search",
        "query",
        "关键词",
        "搜索关键词",
        "检索关键词",
        "简洁关键词",
        "题目中的技术名词和限定词",
        "技术名词和限定词",
        "题目中的真实技术名词",
        "真实技术名词",
        "答案",
        "占位文字",
        "…",
    }
)
_GENERIC_SEARCH_PREFIXES = (
    "题目中的技术名词和限定词",
    "技术名词和限定词",
    "题目中的真实技术名词",
    "真实技术名词",
    "搜索关键词",
    "检索关键词",
    "简洁关键词",
    "关键词",
)


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    heading: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: DocumentChunk
    score: float


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)
    tokens: list[str] = []
    for token in _LATIN_TOKEN.findall(normalized):
        tokens.append(token)
        # Keep exact technical identifiers while also matching punctuation
        # variants such as HF/HNO3 vs. "HF : HNO3" in source material.
        parts = re.findall(r"[a-z0-9]+", token)
        if len(parts) > 1:
            tokens.extend(parts)
    for run in _CJK_RUN.findall(normalized):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _technical_identifiers(text: str) -> set[str]:
    identifiers = {
        _normalize(token.strip("._-"))
        for token in _MIXED_IDENTIFIER.findall(text)
        if re.search(r"[a-z]", token, re.IGNORECASE) and re.search(r"\d", token)
    }
    identifiers.update(_normalize(token) for token in _UPPER_IDENTIFIER.findall(text))
    return identifiers


def _is_substantive_boxed_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    normalized = re.sub(r"[\s，,。.!！?？:：;；…]+", "", _normalize(answer))
    return bool(normalized) and normalized not in {"answer", "答案", "占位文字"}


def extract_search_query(text: str) -> str | None:
    # Base models often mention an opening ``<search>`` token in their
    # reasoning before emitting the actual tool call. Pair the final closing
    # tag with its nearest opening tag instead of letting a regex span both.
    lowered = text.lower()
    close_at = lowered.rfind("</search>")
    if close_at < 0:
        return None
    open_at = lowered.rfind("<search>", 0, close_at)
    if open_at < 0:
        return None
    query = text[open_at + len("<search>") : close_at].strip()
    return query or None


def _question_stem(question: str) -> str:
    match = _QUESTION.search(question)
    stem = match.group(1) if match else question
    return re.sub(r"\s+", " ", stem).strip()


def _is_generic_search_query(query: str) -> bool:
    normalized = re.sub(r"[\s，,。.!！？?：:；;|/\\…]+", "", _normalize(query))
    return not normalized or normalized in _GENERIC_SEARCH_QUERIES


def _strip_generic_search_prefix(query: str) -> str:
    cleaned = query.strip()
    normalized = _normalize(cleaned)
    for prefix in _GENERIC_SEARCH_PREFIXES:
        if normalized.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            return re.sub(r"^[\s，,。.!！？?：:；;|/\\…-]+", "", cleaned).strip()
    return cleaned


def resolve_search_query(query: str, question: str, *, max_chars: int = 300) -> str:
    """Turn a model tool call into a focused retrieval query.

    Base models sometimes copy a prompt placeholder verbatim. In that case the
    question stem is a substantially safer fallback. For a real model query we
    still append the stem, which anchors short or ambiguous terms to the task.
    """
    stem = _question_stem(question)
    focused_query = _strip_generic_search_prefix(query)
    if _is_generic_search_query(focused_query):
        resolved = stem
    elif stem and _normalize(stem) not in _normalize(focused_query):
        resolved = f"{focused_query} {stem}"
    else:
        resolved = focused_query
    return resolved[:max_chars].strip()


def _last_assistant_text(message_log: list[dict[str, Any]]) -> str:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


class MarkdownSearchIndex:
    def __init__(
        self,
        docs_dir: str | Path,
        *,
        chunk_chars: int = 1000,
        overlap_chars: int = 150,
        max_file_bytes: int = 2_000_000,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        if overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("overlap_chars must satisfy 0 <= overlap < chunk_chars")

        self.docs_dir = Path(docs_dir).expanduser().resolve()
        if not self.docs_dir.is_dir():
            raise FileNotFoundError(f"document directory not found: {self.docs_dir}")

        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_file_bytes = max_file_bytes
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self._term_frequencies: list[Counter[str]] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_lengths: list[int] = []
        self._normalized_chunks: list[str] = []
        self._files = 0
        self._characters = 0

        self._load()
        if not self.chunks:
            raise ValueError(f"no readable markdown documents under {self.docs_dir}")
        self._build_index()

    def _load(self) -> None:
        for path in sorted(self.docs_dir.rglob("*.md")):
            if not path.is_file() or path.stat().st_size > self.max_file_bytes:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            self._files += 1
            self._characters += len(text)
            source = path.relative_to(self.docs_dir).as_posix()
            self.chunks.extend(self._split_document(source, text))

    def _split_document(self, source: str, text: str) -> list[DocumentChunk]:
        sections: list[tuple[str, str]] = []
        heading = Path(source).stem
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append((heading, content))

        for line in text.splitlines():
            match = _HEADING.match(line)
            if match:
                flush()
                body = []
                heading = match.group(1).strip()
            else:
                body.append(line)
        flush()

        chunks: list[DocumentChunk] = []
        for section_heading, content in sections:
            combined = f"{section_heading}\n{content}".strip()
            start = 0
            while start < len(combined):
                end = min(start + self.chunk_chars, len(combined))
                if end < len(combined):
                    boundary = combined.rfind("\n", start + self.chunk_chars // 2, end)
                    if boundary > start:
                        end = boundary
                piece = combined[start:end].strip()
                if piece:
                    chunks.append(DocumentChunk(source, section_heading, piece))
                if end >= len(combined):
                    break
                start = max(start + 1, end - self.overlap_chars)
        return chunks

    def _build_index(self) -> None:
        for chunk_id, chunk in enumerate(self.chunks):
            frequencies = Counter(_tokens(chunk.text))
            self._term_frequencies.append(frequencies)
            length = sum(frequencies.values())
            self._doc_lengths.append(length)
            self._normalized_chunks.append(_normalize(chunk.text))
            for term, frequency in frequencies.items():
                self._postings[term].append((chunk_id, frequency))
        self._avg_doc_length = sum(self._doc_lengths) / max(1, len(self._doc_lengths))

    @property
    def stats(self) -> dict[str, int | str]:
        return {
            "docs_dir": str(self.docs_dir),
            "files": self._files,
            "chunks": len(self.chunks),
            "characters": self._characters,
        }

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        query = query.strip()[:300]
        query_terms = Counter(_tokens(query))
        if not query_terms or top_k <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        corpus_size = len(self.chunks)
        for term, query_frequency in query_terms.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            query_weight = 1.0 + math.log(query_frequency)
            for chunk_id, term_frequency in postings:
                length_norm = 1 - self.b + self.b * (
                    self._doc_lengths[chunk_id] / max(1.0, self._avg_doc_length)
                )
                term_score = (
                    term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + self.k1 * length_norm)
                )
                scores[chunk_id] += inverse_document_frequency * term_score * query_weight

        normalized_query = re.sub(r"\s+", "", _normalize(query))
        if len(normalized_query) >= 4:
            for chunk_id in list(scores):
                compact_chunk = re.sub(r"\s+", "", self._normalized_chunks[chunk_id])
                if normalized_query in compact_chunk:
                    scores[chunk_id] += 4.0

        identifiers = _technical_identifiers(query)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        hits: list[SearchHit] = []
        seen_bodies: set[str] = set()
        for chunk_id, score in ranked:
            normalized_chunk = self._normalized_chunks[chunk_id]
            if identifiers and not all(
                identifier in normalized_chunk for identifier in identifiers
            ):
                continue
            chunk = self.chunks[chunk_id]
            body = chunk.text.split("\n", 1)[-1]
            body_signature = re.sub(r"\s+", "", _normalize(body))
            if body_signature in seen_bodies:
                continue
            seen_bodies.add(body_signature)
            hits.append(SearchHit(chunk, score))
            if len(hits) >= top_k:
                break
        return hits

    def format_results(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        max_chars: int = 2600,
    ) -> str:
        if not hits:
            return (
                "<search_results>\n"
                f"查询：{query}\n没有找到匹配资料，请改用更具体的关键词或直接作答。\n"
                "</search_results>"
            )

        parts = ["<search_results>\n", f"查询：{query}"]
        used = sum(len(part) for part in parts)
        for rank, hit in enumerate(hits, 1):
            prefix = f"\n[{rank}] 来源：{hit.chunk.source} | {hit.chunk.heading}\n"
            closing_chars = len("\n</search_results>")
            remaining = max_chars - used - len(prefix) - closing_chars
            if remaining <= 80:
                break
            hits_left = len(hits) - rank + 1
            snippet_budget = max(80, remaining // hits_left)
            snippet = self._focused_snippet(
                hit.chunk.text, query, min(600, snippet_budget)
            )
            parts.append(prefix + snippet)
            used += len(prefix) + len(snippet)
        parts.append("\n</search_results>")
        return "".join(parts)

    @staticmethod
    def _focused_snippet(text: str, query: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        normalized_text = _normalize(text)
        candidates = {term for term in _tokens(query) if len(term) >= 2}

        match_at = -1
        for term in sorted(candidates, key=len, reverse=True):
            match_at = normalized_text.find(term)
            if match_at >= 0:
                break

        if match_at < 0:
            return text[:max_chars].rstrip()

        start = max(0, match_at - max_chars // 4)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "…" + snippet[1:]
        if end < len(text):
            snippet = snippet[:-1] + "…"
        return snippet


class QASearchRunner:
    def __init__(
        self,
        index: MarkdownSearchIndex,
        *,
        reward_fn: Callable[..., list[float]],
        boxed_extractor: Callable[[str], str | None],
        top_k: int = 3,
        max_result_chars: int = 2600,
        max_searches: int = 2,
        max_turns: int = 3,
        format_penalty: float = -0.5,
        first_search_reward: float = 0.0,
    ) -> None:
        self.index = index
        self.reward_fn = reward_fn
        self.boxed_extractor = boxed_extractor
        self.top_k = top_k
        self.max_result_chars = max_result_chars
        self.max_searches = max_searches
        self.max_turns = max_turns
        self.format_penalty = format_penalty
        self.first_search_reward = first_search_reward

    def process_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> tuple[dict[str, str], float, bool, list[str] | None, dict[str, Any] | None, str | None]:
        completion = _last_assistant_text(message_log)
        next_metadata = dict(metadata)
        next_metadata["num_turns"] = int(metadata.get("num_turns", 0)) + 1
        current_turn = next_metadata["num_turns"]

        boxed_answer = self.boxed_extractor(completion)
        if _is_substantive_boxed_answer(boxed_answer):
            reward = self.reward_fn(
                [str(metadata.get("query", ""))],
                [completion],
                [str(metadata.get("expected_answer", ""))],
            )[0]
            return (
                {"role": "environment", "content": "<final>答案已提交。</final>"},
                float(reward),
                True,
                None,
                None,
                boxed_answer,
            )

        search_query = extract_search_query(completion)
        searches = int(metadata.get("num_searches", 0))
        if search_query is not None:
            if current_turn >= self.max_turns:
                return (
                    {
                        "role": "environment",
                        "content": "<error>已达到最大轮数，但尚未提交 \\boxed{...} 答案。</error>",
                    },
                    self.format_penalty,
                    True,
                    None,
                    None,
                    None,
                )
            if searches >= self.max_searches:
                return (
                    {
                        "role": "environment",
                        "content": "<error>检索次数已用完，请根据已有资料直接用 \\boxed{...} 作答。</error>",
                    },
                    0.0,
                    False,
                    ["</search>"],
                    next_metadata,
                    None,
                )
            resolved_query = resolve_search_query(
                search_query, str(metadata.get("query", ""))
            )
            hits = self.index.search(resolved_query, self.top_k)
            next_metadata["num_searches"] = searches + 1
            observation = self.index.format_results(
                resolved_query, hits, max_chars=self.max_result_chars
            )
            if _is_generic_search_query(search_query):
                observation += "\n检索词过于宽泛，系统已改用题目内容检索。"
            observation += (
                "\n请基于上述真实内部资料判断；信息足够时立即用 \\boxed{...} "
                "提交答案，不要继续检索。"
            )
            return (
                {"role": "environment", "content": observation},
                self.first_search_reward if searches == 0 and hits else 0.0,
                False,
                ["</search>"],
                next_metadata,
                None,
            )

        if current_turn >= self.max_turns:
            return (
                {
                    "role": "environment",
                    "content": "<error>输出格式无效，且已达到最大轮数。</error>",
                },
                self.format_penalty,
                True,
                None,
                None,
                None,
            )

        return (
            {
                "role": "environment",
                "content": (
                    "<error>请只选择一种操作：用 <search>关键词</search> 检索，"
                    "或用 \\boxed{答案} 提交最终答案。</error>"
                ),
            },
            0.0,
            False,
            ["</search>"],
            next_metadata,
            None,
        )
