"""Unit tests for fixed-size and semantic chunkers."""

from __future__ import annotations

import pytest

from app.services.chunker import (
    BaseChunker,
    FixedSizeChunker,
    SemanticRecursiveChunker,
    build_chunker,
)


class TestFixedSizeChunker:
    def test_chunks_are_bounded_and_ordered(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=2)
        chunks = chunker.chunk("abcdefghij")

        assert chunks == ["abcde", "defgh", "ghij"]
        assert all(len(chunk) <= 5 for chunk in chunks)

    def test_overlap_is_preserved_between_windows(self) -> None:
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("a" * 1_200)

        assert [len(chunk) for chunk in chunks] == [500, 500, 300]
        assert chunks[0][-50:] == chunks[1][:50]
        assert chunks[1][-50:] == chunks[2][:50]

    def test_whitespace_only_text_produces_no_chunks(self) -> None:
        assert FixedSizeChunker().chunk("   \n\t  ") == []

    @pytest.mark.parametrize(
        ("size", "overlap", "message"),
        [
            (0, 0, "chunk_size"),
            (-1, 0, "chunk_size"),
            (10, -1, "chunk_overlap"),
            (10, 10, "chunk_overlap"),
        ],
    )
    def test_invalid_configuration_is_rejected(
        self, size: int, overlap: int, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)


class TestSemanticRecursiveChunker:
    def test_short_text_is_one_chunk(self) -> None:
        chunker = SemanticRecursiveChunker(max_chunk_size=100)
        assert chunker.chunk("A short paragraph.") == ["A short paragraph."]

    def test_paragraphs_are_packed_within_the_limit(self) -> None:
        chunker = SemanticRecursiveChunker(max_chunk_size=60)
        text = "First policy paragraph.\n\nSecond policy paragraph.\n\nThird policy paragraph."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        assert all(len(chunk) <= 60 for chunk in chunks)
        assert "First policy paragraph." in chunks[0]

    def test_long_sentence_is_hard_split_at_word_boundaries(self) -> None:
        chunker = SemanticRecursiveChunker(max_chunk_size=25)
        text = "word " * 30
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all(0 < len(chunk) <= 25 for chunk in chunks)
        assert all(not chunk.startswith(" ") for chunk in chunks)

    def test_paragraph_and_sentence_boundaries_are_used(self) -> None:
        chunker = SemanticRecursiveChunker(max_chunk_size=40)
        chunks = chunker.chunk(
            "Alpha beta gamma. Delta epsilon zeta.\n\nEta theta iota. Kappa lambda mu."
        )

        assert all(len(chunk) <= 40 for chunk in chunks)
        assert any("Alpha" in chunk for chunk in chunks)
        assert any("Eta" in chunk for chunk in chunks)

    def test_crlf_is_normalised(self) -> None:
        chunker = SemanticRecursiveChunker(max_chunk_size=80)
        chunks = chunker.chunk("First line.\r\n\r\nSecond line.")
        assert chunks == ["First line.\n\nSecond line."]

    def test_whitespace_only_text_produces_no_chunks(self) -> None:
        assert SemanticRecursiveChunker().chunk("  \r\n  ") == []

    def test_invalid_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_chunk_size"):
            SemanticRecursiveChunker(max_chunk_size=0)


class TestBuildChunker:
    def test_fixed_strategy(self) -> None:
        chunker = build_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)
        assert isinstance(chunker, BaseChunker)

    def test_semantic_strategy(self) -> None:
        chunker = build_chunker("semantic")
        assert isinstance(chunker, SemanticRecursiveChunker)
        assert isinstance(chunker, BaseChunker)

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            build_chunker("magic")
