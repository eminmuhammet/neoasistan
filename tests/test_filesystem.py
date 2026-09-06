import asyncio
from unittest.mock import patch

from neo.tools.filesystem import FindFileTool, OpenFolderTool


def test_find_file_matches_by_name_fragment(tmp_path):
    (tmp_path / "aylik_rapor.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "notlar.txt").write_text("x", encoding="utf-8")

    result = asyncio.run(FindFileTool().run(query="rapor", location=str(tmp_path)))

    assert result.success
    assert [m["name"] for m in result.data["matches"]] == ["aylik_rapor.pdf"]


def test_find_file_matches_by_extension(tmp_path):
    (tmp_path / "a.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")

    result = asyncio.run(FindFileTool().run(query=".pdf", location=str(tmp_path)))

    assert result.data["count"] == 2


def test_find_file_searches_subfolders(tmp_path):
    nested = tmp_path / "alt" / "daha_alt"
    nested.mkdir(parents=True)
    (nested / "gizli_rapor.docx").write_text("x", encoding="utf-8")

    result = asyncio.run(FindFileTool().run(query="gizli", location=str(tmp_path)))

    assert result.data["count"] == 1


def test_find_file_reports_unknown_location():
    result = asyncio.run(FindFileTool().run(query="x", location="olmayan-klasor-12345"))
    assert not result.success


def test_find_file_requires_a_query(tmp_path):
    result = asyncio.run(FindFileTool().run(query="   ", location=str(tmp_path)))
    assert not result.success


def test_find_file_returns_empty_list_when_nothing_matches(tmp_path):
    (tmp_path / "baska.txt").write_text("x", encoding="utf-8")
    result = asyncio.run(FindFileTool().run(query="bulunmaz", location=str(tmp_path)))
    assert result.success
    assert result.data["matches"] == []


def test_open_folder_opens_resolved_path(tmp_path):
    with patch("neo.tools.filesystem.os.startfile") as mock_start:
        result = asyncio.run(OpenFolderTool().run(folder=str(tmp_path)))

    assert result.success
    mock_start.assert_called_once_with(str(tmp_path))


def test_open_folder_accepts_turkish_known_folder_names():
    with patch("neo.tools.filesystem.os.startfile") as mock_start:
        result = asyncio.run(OpenFolderTool().run(folder="indirilenler"))

    assert result.success
    assert "Downloads" in mock_start.call_args[0][0]


def test_open_folder_reports_unknown_folder():
    result = asyncio.run(OpenFolderTool().run(folder="boyle-bir-klasor-yok-999"))
    assert not result.success
