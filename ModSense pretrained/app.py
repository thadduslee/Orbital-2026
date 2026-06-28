import os
import re
import threading
import time

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

app = Flask(__name__)

_state = {
    "model": "",
    "tokenizer": "",
    "df_reviews": pd.DataFrame(),
    "ready": False,
    "loading": False,
    "error": "",
    "progress": "",
}
_lock = threading.Lock()

# Model config
BASE_MODEL_ID = "microsoft/deberta-v3-base"
ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "")
NUM_LABELS = 5
MIN_SAMPLES = 5

# Disqus config
FORUM_NAME = "nusmods-prod"
LIMIT = 100
DELAY = 1.0


def load_hf_model():
    if not ADAPTER_REPO:
        raise RuntimeError(
            "HF_ADAPTER_REPO environment variable is not set. "
            "Set it to your HuggingFace adapter repo, e.g. "
            "'your-username/deberta-lora-module-scorer'."
        )

    _state["progress"] = f"Loading base model '{BASE_MODEL_ID}' from HuggingFace..."
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=NUM_LABELS,
    )

    _state["progress"] = f"Applying LoRA adapters from '{ADAPTER_REPO}'..."
    model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
    model.eval()

    _state["progress"] = "Model ready."
    return tokenizer, model


def _api_key() -> str:
    key = os.environ.get("DISQUS_API_KEY", "")
    if not key:
        raise RuntimeError("DISQUS_API_KEY environment variable is not set.")
    return key


def clean_review_text(html_text: str) -> str:
    if not html_text:
        return ""

    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator=" ")

    text = re.sub(r"Module review by.*?:", "", text)
    text = re.sub(r"Taken in AY\d+/\d+ Sem \d+", "", text)
    text = re.sub(r"Module review also posted here:.*", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def module_code_from_title(title: str) -> str:
    if not title:
        return "UNKNOWN"
    first = title.strip().split()[0]
    return first if re.match(r"^[A-Z]{2,4}\d{4}[A-Z]?$", first) else "UNKNOWN"


def fetch_all(endpoint: str, label: str, extra: dict | None = None):
    key = _api_key()
    url = f"https://disqus.com/api/3.0/{endpoint}"
    items = []
    cursor = ""

    while True:
        params = {
            "api_key": key,
            "forum": FORUM_NAME,
            "limit": LIMIT,
            "order": "asc",
        }
        if extra:
            params.update(extra)
        if cursor:
            params["cursor"] = cursor

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2**attempt)
                continue

            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 30)))
                continue

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            break
        else:
            raise RuntimeError("Max retries exceeded")

        body = resp.json()
        if body.get("code", 0) != 0:
            raise RuntimeError(f"Disqus error {body.get('code')}: {body.get('response')}")

        batch = body.get("response", [])
        items.extend(batch)
        _state["progress"] = f"Fetching {label}... {len(items)} so far"

        cursor_data = body.get("cursor", {})
        if cursor_data.get("hasNext"):
            cursor = cursor_data["next"]
            time.sleep(DELAY)
        else:
            break

    return items


def load_disqus_data():
    _state["progress"] = "Verifying Disqus forum..."
    key = _api_key()
    resp = requests.get(
        "https://disqus.com/api/3.0/forums/details.json",
        params={"api_key": key, "forum": FORUM_NAME},
        timeout=15,
    )
    info = resp.json()
    if info.get("code", -1) != 0:
        raise RuntimeError(f"Disqus error {info.get('code')}: check your API key.")

    threads = fetch_all("forums/listThreads.json", "threads")
    thread_map = {
        t["id"]: {
            "module_code": module_code_from_title(t.get("title", "")),
            "module_title": t.get("title", "Unknown"),
        }
        for t in threads
    }

    raw_posts = fetch_all("forums/listPosts.json", "posts", extra={"include": "approved"})
    records = []

    for post in raw_posts:
        tid = post.get("thread")
        info = thread_map.get(tid, {"module_code": "UNKNOWN", "module_title": "Unknown"})
        raw = post.get("message", "") or ""
        records.append(
            {
                "module_code": info["module_code"],
                "module_title": info["module_title"],
                "date": post.get("createdAt"),
                "message": clean_review_text(raw),
            }
        )

    df = pd.DataFrame(records)
    df = df[
        (df["module_code"] != "UNKNOWN")
        & (df["message"].str.strip() != "")
    ].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return df


def predict_scores(tokenizer, model, messages: list[str]) -> list[float]:
    scores = []
    if len(messages) == 0:
        return scores

    with torch.no_grad():
        for msg in messages:
            inputs = tokenizer(
                msg,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()
            score = float((probs * torch.arange(NUM_LABELS, dtype=torch.float)).sum()) + 1.0
            scores.append(score)

    return scores


def sentiment_from_average(average: float) -> str:
    if average > 3.5:
        return "positive"
    if average < 2.5:
        return "negative"
    return "neutral"


def get_academic_period(date_value) -> str:
    date = pd.to_datetime(date_value, errors="coerce", utc=True)
    if pd.isna(date):
        return ""

    month = date.month
    year = date.year

    if month >= 8:
        return f"AY{year}/{year + 1} Sem 1"
    if month <= 5:
        return f"AY{year - 1}/{year} Sem 2"
    return f"AY{year - 1}/{year} Special Term"


def period_sort_key(period: str):
    term_order = {"Sem 1": 0, "Sem 2": 1, "Special Term": 2}
    try:
        ay_part, term = period.split(" ", 1)
        start_year = int(ay_part[2:6])
        return (start_year, term_order.get(term, 99))
    except Exception:
        return (9999, 99)


def build_semester_sentiment(subset: pd.DataFrame, scores: list[float]) -> list[dict]:
    if not scores:
        return []

    scored = subset.copy().reset_index(drop=True)
    scored["sentiment_score"] = scores
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce", utc=True)
    scored = scored.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if scored.empty:
        return []

    scored["period"] = scored["date"].apply(get_academic_period)
    scored = scored[scored["period"] != ""].reset_index(drop=True)

    if scored.empty:
        return []

    scored["cum_n"] = np.arange(1, len(scored) + 1)
    scored["cum_avg"] = scored["sentiment_score"].expanding().mean()

    rows = []
    for period, group in scored.groupby("period", sort=False):
        period_review_count = int(len(group))
        period_average = round(float(group["sentiment_score"].mean()), 2)
        cumulative_review_count = int(group["cum_n"].iloc[-1])
        cumulative_average = round(float(group["cum_avg"].iloc[-1]), 2)

        rows.append(
            {
                "period": period,
                "period_review_count": period_review_count,
                "period_average": period_average,
                "period_sentiment": sentiment_from_average(period_average),
                "cumulative_review_count": cumulative_review_count,
                "cumulative_average": cumulative_average,
                "cumulative_sentiment": sentiment_from_average(cumulative_average),
                "reliable": cumulative_review_count >= MIN_SAMPLES,
            }
        )

    return sorted(rows, key=lambda row: period_sort_key(row["period"]))


def _background_init():
    with _lock:
        _state["loading"] = True
        _state["error"] = ""

    try:
        tokenizer, model = load_hf_model()
        with _lock:
            _state["tokenizer"] = tokenizer
            _state["model"] = model

        df = load_disqus_data()
        with _lock:
            _state["df_reviews"] = df
            _state["ready"] = True
            _state["progress"] = (
                f"Ready, {len(df)} reviews across {df['module_code'].nunique()} modules."
            )
    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)
            _state["progress"] = f"Error: {exc}"
    finally:
        with _lock:
            _state["loading"] = False


def start_init():
    thread = threading.Thread(target=_background_init, daemon=True)
    thread.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify(
            {
                "ready": _state["ready"],
                "loading": _state["loading"],
                "error": _state["error"],
                "progress": _state["progress"],
            }
        )


@app.route("/api/modules")
def api_modules():
    with _lock:
        if not _state["ready"]:
            return jsonify({"error": "Data not ready yet."}), 503
        modules = sorted(_state["df_reviews"]["module_code"].unique().tolist())

    return jsonify({"modules": modules, "count": len(modules)})


@app.route("/api/analyze/<module_code>")
def api_analyze(module_code: str):
    try:
        module_code = module_code.upper().strip()

        with _lock:
            if not _state["ready"]:
                return jsonify({"error": "Data not ready yet."}), 503
            model = _state["model"]
            tokenizer = _state["tokenizer"]
            df = _state["df_reviews"]

        subset = df[df["module_code"] == module_code].copy().reset_index(drop=True)
        if subset.empty:
            close = sorted(
                {m for m in df["module_code"].unique() if m.startswith(module_code[:2])}
            )[:8]
            return jsonify(
                {
                    "found": False,
                    "module_code": module_code,
                    "suggestions": close,
                }
            ), 404

        messages = subset["message"].tolist()
        review_count = len(messages)
        
        scores = predict_scores(tokenizer, model, messages) 

        average = round(sum(scores) / review_count, 2) if review_count else 0.0
        reliable = review_count >= MIN_SAMPLES
        sentiment = sentiment_from_average(average) if reliable else "insufficient"
        semester_sentiment = build_semester_sentiment(subset, scores)

        response = {
            "found": True,
            "module_code": module_code,
            "module_title": subset["module_title"].iloc[0],
            "review_count": review_count,
            "min_samples": MIN_SAMPLES,
            "reliable": reliable,
            "score": average if reliable else 0.0,
            "raw_score": average,
            "sentiment": sentiment,
            "message": "" if reliable else f"Not enough reviews to analyse sentiment for {module_code}",
            "score_distribution": {
                "positive": sum(1 for s in scores if s > 3.5),
                "neutral": sum(1 for s in scores if 2.5 <= s <= 3.5),
                "negative": sum(1 for s in scores if s < 2.5),
            },
            "semester_sentiment": semester_sentiment,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


def _background_reload_data():
    with _lock:
        _state["loading"] = True
        _state["error"] = ""
        _state["ready"] = False

    try:
        df = load_disqus_data()
        with _lock:
            _state["df_reviews"] = df
            _state["ready"] = True
            _state["progress"] = (
                f"Ready, {len(df)} reviews across {df['module_code'].nunique()} modules."
            )
    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)
            _state["progress"] = f"Error: {exc}"
    finally:
        with _lock:
            _state["loading"] = False


@app.route("/api/reload", methods=["POST"])
def api_reload():
    with _lock:
        if _state["loading"]:
            return jsonify({"error": "Already loading."}), 409

    thread = threading.Thread(target=_background_reload_data, daemon=True)
    thread.start()
    return jsonify({"message": "Data reload started (model unchanged)."})


start_init()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
