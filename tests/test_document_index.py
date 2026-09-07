from neo.memory.document_index import DocumentIndex, _tokenize


def _index(tmp_path) -> DocumentIndex:
    return DocumentIndex(tmp_path / "index.db")


def test_tokenize_folds_turkish_diacritics():
    assert _tokenize("Yapay Zekâ Bağımsızlığı") == ["yapay", "zeka", "bagimsizligi"]


def test_upsert_and_list_documents(tmp_path):
    index = _index(tmp_path)
    index.upsert("/a.txt", "a.txt", "merhaba dünya")

    docs = index.list_documents()

    assert len(docs) == 1
    assert docs[0].name == "a.txt"


def test_upsert_same_path_replaces_content(tmp_path):
    index = _index(tmp_path)
    index.upsert("/a.txt", "a.txt", "eski içerik")
    index.upsert("/a.txt", "a.txt", "yeni içerik")

    assert index.count() == 1
    hits = index.search("yeni")
    assert len(hits) == 1


def test_remove_deletes_a_document(tmp_path):
    index = _index(tmp_path)
    index.upsert("/a.txt", "a.txt", "içerik")

    assert index.remove("/a.txt") is True
    assert index.count() == 0


def test_remove_missing_path_reports_nothing_removed(tmp_path):
    index = _index(tmp_path)
    assert index.remove("/nope.txt") is False


def test_search_with_no_documents_returns_empty(tmp_path):
    index = _index(tmp_path)
    assert index.search("herhangi bir şey") == []


def test_search_with_empty_query_returns_empty(tmp_path):
    index = _index(tmp_path)
    index.upsert("/a.txt", "a.txt", "içerik")
    assert index.search("") == []


def test_search_ranks_the_more_relevant_document_first(tmp_path):
    index = _index(tmp_path)
    index.upsert(
        "/ai.txt", "ai.txt",
        "yapay zeka bagimsizligi konusu bu raporun ana temasidir. "
        "yapay zeka bagimsizligi her yerde geciyor.",
    )
    index.upsert("/other.txt", "other.txt", "bu dosya tamamen farkli bir konudan bahsediyor, tatil planlari.")

    hits = index.search("yapay zeka bagimsizligi")

    assert hits[0].name == "ai.txt"
    assert hits[0].score > 0


def test_search_returns_a_relevant_snippet(tmp_path):
    index = _index(tmp_path)
    long_prefix = "dolgu metni " * 50
    index.upsert("/a.txt", "a.txt", long_prefix + "burada onemli konu hakkinda detay var" + " dolgu" * 50)

    hits = index.search("onemli konu")

    assert hits
    assert "onemli konu" in hits[0].snippet.lower()


def test_search_respects_top_k(tmp_path):
    index = _index(tmp_path)
    for i in range(10):
        index.upsert(f"/doc{i}.txt", f"doc{i}.txt", "ortak kelime " * 5)

    hits = index.search("ortak kelime", top_k=3)

    assert len(hits) == 3


def test_search_ignores_documents_with_no_matching_terms(tmp_path):
    index = _index(tmp_path)
    index.upsert("/a.txt", "a.txt", "tamamen alakasiz bir metin")

    hits = index.search("bulunmayan kelime grubu")

    assert hits == []
