"""Chunking strategies: fixed-size windows and recursive semantic splitting."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.services.extractor import collapse_internal_whitespace


class BaseChunker(ABC):
    """Strategy interface for splitting document text into chunks."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split ``text`` into an ordered list of non-empty chunks."""


class FixedSizeChunker(BaseChunker):
    """Sliding-window chunker with a fixed character window and overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        normalised = text.strip()
        if not normalised:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        for start in range(0, len(normalised), step):
            piece = normalised[start : start + self.chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + self.chunk_size >= len(normalised):
                break
        return chunks


class SemanticRecursiveChunker(BaseChunker):
    """Paragraph/sentence-aware chunker bounded by ``max_chunk_size`` characters.

    Splitting follows the document structure: paragraphs (``\\n\\n``) first, then
    sentences (``. `` / ``! `` / ``? ``), and finally word-boundary hard splits
    for any single sentence that exceeds the limit.
    """

    _PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")
    _SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 800) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero.")
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        normalised = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalised:
            return []

        units: list[str] = []
        for paragraph in self._PARAGRAPH_PATTERN.split(normalised):
            units.extend(self._split_paragraph(paragraph))

        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = unit if not current else f"{current}\n\n{unit}"
            if len(candidate) <= self.max_chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = unit
        if current:
            chunks.append(current)
        return chunks

    def _split_paragraph(self, paragraph: str) -> list[str]:
        cleaned = collapse_internal_whitespace(paragraph)
        if not cleaned:
            return []
        if len(cleaned) <= self.max_chunk_size:
            return [cleaned]

        sentences = [
            sentence.strip()
            for sentence in self._SENTENCE_PATTERN.split(cleaned)
            if sentence.strip()
        ]

        segments: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = sentence if not current else f"{current} {sentence}"
            if len(candidate) <= self.max_chunk_size:
                current = candidate
                continue

            if current:
                segments.append(current)
                current = ""

            if len(sentence) <= self.max_chunk_size:
                current = sentence
            else:
                pieces = self._hard_split(sentence)
                segments.extend(pieces[:-1])
                current = pieces[-1]

        if current:
            segments.append(current)
        return segments

    def _hard_split(self, sentence: str) -> list[str]:
        limit = self.max_chunk_size
        pieces: list[str] = []
        start = 0

        while start < len(sentence):
            end = min(start + limit, len(sentence))
            if end < len(sentence):
                boundary = sentence.rfind(" ", start, end)
                if boundary > start + limit // 2:
                    end = boundary

            piece = sentence[start:end].strip()
            if piece:
                pieces.append(piece)

            start = end
            while start < len(sentence) and sentence[start].isspace():
                start += 1

        return pieces


def build_chunker(strategy: str) -> BaseChunker:
    """Return the chunker implementing ``strategy`` (``fixed`` or ``semantic``)."""
    match strategy:
        case "fixed":
            return FixedSizeChunker()
        case "semantic":
            return SemanticRecursiveChunker()
        case _:
            raise ValueError(f"Unknown chunking strategy: {strategy!r}.")
