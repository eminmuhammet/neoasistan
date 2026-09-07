import asyncio

from neo.memory.document_index import DocumentIndex
from neo.tools.base import RiskLevel
from neo.tools.document_search import IndexFolderTool, SearchDocumentsTool


def _index(tmp_path) -> DocumentIndex:
    return DocumentIndex(tmp_path / "index.db")


def test_index_folder_indexes_supported_files(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("yapay zeka hakkinda notlar", encoding="utf-8")
    (folder / "b.md").write_text("baska bir konu", encoding="utf-8")
    (folder / "ignore.exe").write_bytes(b"\x00\x01")
    index = _index(tmp_path)

    result = asyncio.run(IndexFolderTool(index).run(folder=str(folder)))

    assert result.success is True
    assert result.data["indexed"] == 2
    assert result.data["skipped"] == 0
    assert index.count() == 2


def test_index_folder_reports_missing_folder(tmp_path):
    index = _index(tmp_path)

    result = asyncio.run(IndexFolderTool(index).run(folder=str(tmp_path / "yok")))

    assert result.success is False


def test_index_folder_counts_unreadable_files_as_skipped(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "broken.pdf").write_bytes(b"not a real pdf")
    index = _index(tmp_path)

    result = asyncio.run(IndexFolderTool(index).run(folder=str(folder)))

    assert result.data["indexed"] == 0
    assert result.data["skipped"] == 1


def test_search_documents_returns_matches_after_indexing(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "ai.txt").write_text(
        "yapay zeka bagimsizligi konusunda hazirladigim komite raporu", encoding="utf-8"
    )
    index = _index(tmp_path)
    asyncio.run(IndexFolderTool(index).run(folder=str(folder)))

    result = asyncio.run(SearchDocumentsTool(index).run(query="yapay zeka bagimsizligi"))

    assert result.success is True
    assert result.data["count"] == 1
    assert "ai.txt" in result.data["results"][0]["name"]


def test_search_documents_rejects_empty_query(tmp_path):
    index = _index(tmp_path)

    result = asyncio.run(SearchDocumentsTool(index).run(query="   "))

    assert result.success is False


def test_search_documents_with_nothing_indexed_returns_empty(tmp_path):
    index = _index(tmp_path)

    result = asyncio.run(SearchDocumentsTool(index).run(query="herhangi bir sey"))

    assert result.success is True
    assert result.data["count"] == 0


def test_index_folder_is_medium_risk_search_is_low_risk():
    assert IndexFolderTool.risk == RiskLevel.MEDIUM
    assert SearchDocumentsTool.risk == RiskLevel.LOW
