from neo.tools.app_registry import resolve_application


def test_resolves_notepad_via_path_or_registry():
    result = resolve_application("notepad")
    assert result is not None


def test_unknown_application_returns_none():
    result = resolve_application("kesinlikle-var-olmayan-bir-uygulama-adi-12345")
    assert result is None
