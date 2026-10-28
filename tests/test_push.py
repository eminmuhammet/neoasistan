from neo.web.push import (
    PushSubscriptionStore,
    load_or_create_vapid_private_key,
    send_push_to_all,
    vapid_public_key_b64,
)


def test_load_or_create_vapid_private_key_is_idempotent(tmp_path):
    path = tmp_path / "vapid_private.pem"

    first = load_or_create_vapid_private_key(path)
    first_bytes = path.read_bytes()
    second = load_or_create_vapid_private_key(path)

    assert first == second == path
    assert path.read_bytes() == first_bytes


def test_vapid_public_key_is_stable_for_the_same_private_key(tmp_path):
    path = tmp_path / "vapid_private.pem"
    load_or_create_vapid_private_key(path)

    key_a = vapid_public_key_b64(path)
    key_b = vapid_public_key_b64(path)

    assert key_a == key_b
    assert len(key_a) > 40


def test_vapid_public_keys_differ_across_installs(tmp_path):
    path_a = tmp_path / "a.pem"
    path_b = tmp_path / "b.pem"
    load_or_create_vapid_private_key(path_a)
    load_or_create_vapid_private_key(path_b)

    assert vapid_public_key_b64(path_a) != vapid_public_key_b64(path_b)


def _store(tmp_path) -> PushSubscriptionStore:
    return PushSubscriptionStore(tmp_path / "push.db")


def test_add_and_list_subscriptions(tmp_path):
    store = _store(tmp_path)
    store.add("https://push.example/a", "p256dh-a", "auth-a")

    subs = store.list_all()

    assert len(subs) == 1
    assert subs[0].endpoint == "https://push.example/a"


def test_add_same_endpoint_twice_updates_rather_than_duplicates(tmp_path):
    store = _store(tmp_path)
    store.add("https://push.example/a", "old-key", "old-auth")
    store.add("https://push.example/a", "new-key", "new-auth")

    subs = store.list_all()

    assert len(subs) == 1
    assert subs[0].p256dh == "new-key"


def test_remove_deletes_a_subscription(tmp_path):
    store = _store(tmp_path)
    store.add("https://push.example/a", "k", "a")

    store.remove("https://push.example/a")

    assert store.list_all() == []


def test_remove_missing_endpoint_does_not_raise(tmp_path):
    store = _store(tmp_path)
    store.remove("https://push.example/does-not-exist")  # must not raise


def test_send_push_to_all_calls_webpush_for_each_subscription(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.add("https://push.example/a", "key-a", "auth-a")
    store.add("https://push.example/b", "key-b", "auth-b")
    calls = []

    def fake_webpush(subscription_info, data, vapid_private_key, vapid_claims, headers=None, **kwargs):
        calls.append((subscription_info["endpoint"], data))
        return "ok"

    import pywebpush
    monkeypatch.setattr(pywebpush, "webpush", fake_webpush)

    sent = send_push_to_all(store, tmp_path / "vapid.pem", "Başlık", "Gövde metni")

    import json

    assert sent == 2
    assert {c[0] for c in calls} == {"https://push.example/a", "https://push.example/b"}
    assert json.loads(calls[0][1]) == {"title": "Başlık", "body": "Gövde metni"}


def test_send_push_removes_expired_subscriptions_on_410(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.add("https://push.example/gone", "k", "a")

    import pywebpush

    class FakeResponse:
        status_code = 410

    def fake_webpush(*args, **kwargs):
        raise pywebpush.WebPushException("gone", response=FakeResponse())

    monkeypatch.setattr(pywebpush, "webpush", fake_webpush)

    sent = send_push_to_all(store, tmp_path / "vapid.pem", "T", "B")

    assert sent == 0
    assert store.list_all() == []


def test_send_push_keeps_subscription_on_transient_error(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.add("https://push.example/still-here", "k", "a")

    import pywebpush

    class FakeResponse:
        status_code = 500

    def fake_webpush(*args, **kwargs):
        raise pywebpush.WebPushException("server error", response=FakeResponse())

    monkeypatch.setattr(pywebpush, "webpush", fake_webpush)

    sent = send_push_to_all(store, tmp_path / "vapid.pem", "T", "B")

    assert sent == 0
    assert len(store.list_all()) == 1
