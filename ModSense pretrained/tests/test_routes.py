import pytest
from unittest.mock import patch
import pandas as pd
import app as app_module
from app import app


class FakeTokenizer:
    """Minimal stand-in for a real HF tokenizer, used only by get_example_reviews'
    token-length filtering. Splitting on whitespace is a crude but sufficient
    stand-in for a token count in tests."""
    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def ready_state():
    """Populate _state directly so tests don't depend on Disqus or the real model."""
    df = pd.DataFrame({
        "module_code": ["CS2103T"] * 6 + ["CS2100"] * 2,
        "module_title": ["Software Engineering"] * 6 + ["Computer Organisation"] * 2,
        "message": ["Great module, learnt a lot"] * 6 + ["Tough but rewarding"] * 2,
        "date": pd.to_datetime(["2024-09-01"] * 8, utc=True),
    })
    app_module._state.update({
        "ready": True, "loading": False, "error": "",
        "df_reviews": df, "model": object(), "tokenizer": FakeTokenizer(),
    })
    yield
    app_module._state.update({"ready": False, "df_reviews": pd.DataFrame()})


def test_status_reports_not_ready_before_init(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "ready" in r.get_json()


def test_modules_returns_503_when_not_ready(client):
    r = client.get("/api/modules")
    assert r.status_code == 503


def test_modules_returns_sorted_codes_when_ready(client, ready_state):
    r = client.get("/api/modules")
    assert r.status_code == 200
    assert r.get_json()["modules"] == ["CS2100", "CS2103T"]


def test_analyze_unknown_module_returns_404_with_suggestions(client, ready_state):
    r = client.get("/api/analyze/CS9999")
    assert r.status_code == 404
    body = r.get_json()
    assert body["found"] is False
    assert "CS2103T" in body["suggestions"]


@patch("app.predict_scores")
def test_analyze_reliable_module_returns_full_result(mock_predict, client, ready_state):
    # side_effect keys the returned scores to len(messages) so this stays correct
    # even if find_module_suggestion triggers additional internal calls.
    mock_predict.side_effect = lambda tokenizer, model, messages: [4.0] * len(messages)
    r = client.get("/api/analyze/CS2103T")
    body = r.get_json()
    assert body["found"] is True
    assert body["reliable"] is True
    assert body["sentiment"] == "positive"
    assert body["review_count"] == 6


@patch("app.predict_scores")
def test_analyze_below_min_samples_flagged_insufficient(mock_predict, client, ready_state):
    mock_predict.side_effect = lambda tokenizer, model, messages: [4.0] * len(messages)
    r = client.get("/api/analyze/CS2100")
    body = r.get_json()
    assert body["reliable"] is False
    assert body["sentiment"] == "insufficient"
    assert body["score"] == 0.0  # score suppressed when unreliable


def test_compare_requires_at_least_two_codes(client, ready_state):
    r = client.get("/api/compare?modules=CS2103T")
    assert r.status_code == 400


@patch("app.predict_scores")
def test_compare_returns_one_entry_per_module(mock_predict, client, ready_state):
    # Each module has a different review count (6 vs 2), so the mock must
    # return a list matching the length of whatever messages it's called with,
    # rather than a fixed-length list.
    mock_predict.side_effect = lambda tokenizer, model, messages: [4.0] * len(messages)
    r = client.get("/api/compare?modules=CS2103T,CS2100")
    body = r.get_json()
    assert len(body["modules"]) == 2
    assert {m["module_code"] for m in body["modules"]} == {"CS2103T", "CS2100"}


def test_reload_returns_409_if_already_loading(client):
    app_module._state["loading"] = True
    r = client.post("/api/reload")
    assert r.status_code == 409
    app_module._state["loading"] = False


def test_disqus_api_key_never_leaks_in_any_response(client, ready_state):
    for endpoint in ["/api/status", "/api/modules", "/api/analyze/CS2103T"]:
        body = client.get(endpoint).get_data(as_text=True)
        assert "DISQUS_API_KEY" not in body
