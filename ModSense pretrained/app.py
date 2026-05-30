"""
NUSMods Sentiment Analysis — Flask Backend
==========================================
The DISQUS_API_KEY is read ONLY from the environment.
It is never forwarded to the frontend.
"""

import os, json, csv, time, re, threading
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv

# ── Load .env (only used locally; ignored when env vars are set by the platform) ──
load_dotenv()

app = Flask(__name__)

_state = {
    "model":        None,   # pretrained Keras model
    "df_reviews":   None,   # pandas DataFrame of all reviews
    "ready":        False,  # True once both model + data are loaded
    "loading":      False,  # True while background init is running
    "error":        None,   # string if init failed
    "progress":     "",     # human-readable status message
}
_lock = threading.Lock()



MODEL_PATH = "sentiment_model.keras"

def load_pretrained_model():
    """Load the pretrained sentiment model bundled with this app."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Pretrained model not found: {MODEL_PATH}")

    _state["progress"] = "Loading pretrained sentiment model..."
    model = tf.keras.models.load_model(MODEL_PATH)
    _state["progress"] = "Pretrained model loaded."
    return model



FORUM_NAME  = "nusmods-prod"
LIMIT       = 100
DELAY       = 1.0

def _api_key() -> str:
    key = os.environ.get("DISQUS_API_KEY", "")
    if not key:
        raise RuntimeError("DISQUS_API_KEY environment variable is not set.")
    return key

def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    text = BeautifulSoup(html_text, "html.parser").get_text(separator=" ")
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

def fetch_all(endpoint: str, label: str, extra: dict = None):
    key   = _api_key()
    url   = f"https://disqus.com/api/3.0/{endpoint}"
    items = []
    cursor = None

    while True:
        params = {"api_key": key, "forum": FORUM_NAME, "limit": LIMIT, "order": "asc"}
        if extra:
            params.update(extra)
        if cursor:
            params["cursor"] = cursor

        for attempt in range(5):
            try:
                resp = requests.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 30)))
                continue
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")
            break
        else:
            raise RuntimeError("Max retries exceeded")

        body  = resp.json()
        batch = body.get("response", [])
        items.extend(batch)
        _state["progress"] = f"Fetching {label}… {len(items)} so far"

        cursor_data = body.get("cursor", {})
        if cursor_data.get("hasNext"):
            cursor = cursor_data["next"]
            time.sleep(DELAY)
        else:
            break

    return items

def load_disqus_data():
    """Fetch threads + posts from Disqus and build df_reviews."""
    _state["progress"] = "Verifying Disqus forum…"
    key = _api_key()

    
    resp = requests.get(
        "https://disqus.com/api/3.0/forums/details.json",
        params={"api_key": key, "forum": FORUM_NAME},
        timeout=15,
    )
    info = resp.json()
    if info.get("code", -1) != 0:
        raise RuntimeError(f"Disqus error {info.get('code')}: check your API key.")

    
    threads   = fetch_all("forums/listThreads.json", "threads")
    thread_map = {
        t["id"]: {
            "module_code":  module_code_from_title(t.get("title", "")),
            "module_title": t.get("title", "Unknown"),
        }
        for t in threads
    }

    
    raw_posts = fetch_all("forums/listPosts.json", "posts", extra={"include": "approved"})

    records = []
    for post in raw_posts:
        tid  = post.get("thread")
        info = thread_map.get(tid, {"module_code": "UNKNOWN", "module_title": "Unknown"})
        raw  = post.get("message", "") or ""
        records.append({
            "module_code":  info["module_code"],
            "module_title": info["module_title"],
            "message":      clean_html(raw),
        })

    df = pd.DataFrame(records)
    df = df[(df["module_code"] != "UNKNOWN") & (df["message"].str.strip() != "")].reset_index(drop=True)
    return df


def _background_init():
    with _lock:
        _state["loading"] = True
        _state["error"]   = None
    try:
        model = load_pretrained_model()
        with _lock:
            _state["model"] = model
        df = load_disqus_data()
        with _lock:
            _state["df_reviews"] = df
            _state["ready"]      = True
            _state["progress"]   = f"Ready — {len(df)} reviews across {df['module_code'].nunique()} modules."
    except Exception as exc:
        with _lock:
            _state["error"]    = str(exc)
            _state["progress"] = f"Error: {exc}"
    finally:
        with _lock:
            _state["loading"] = False

def start_init():
    t = threading.Thread(target=_background_init, daemon=True)
    t.start()



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "ready":    _state["ready"],
            "loading":  _state["loading"],
            "error":    _state["error"],
            "progress": _state["progress"],
        })


@app.route("/api/modules")
def api_modules():
    with _lock:
        if not _state["ready"]:
            return jsonify({"error": "Data not ready yet."}), 503
        modules = sorted(_state["df_reviews"]["module_code"].unique().tolist())
    return jsonify({"modules": modules, "count": len(modules)})


@app.route("/api/analyze/<module_code>")
def api_analyze(module_code: str):
    module_code = module_code.upper().strip()
    with _lock:
        if not _state["ready"]:
            return jsonify({"error": "Data not ready yet."}), 503
        model     = _state["model"]
        df        = _state["df_reviews"]

    subset = df[df["module_code"] == module_code]
    if subset.empty:
        # Try to suggest close matches
        close = sorted({m for m in df["module_code"].unique() if m.startswith(module_code[:2])})[:8]
        return jsonify({
            "found":       False,
            "module_code": module_code,
            "suggestions": close,
        }), 404

    messages = subset["message"].tolist()
    scores   = [float(model.predict(tf.constant([m]), verbose=0)[0][0]) for m in messages]
    average  = round(sum(scores) / len(scores), 2)

    if average > 3.5:
        sentiment = "positive"
    elif average < 2.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return jsonify({
        "found":        True,
        "module_code":  module_code,
        "module_title": subset["module_title"].iloc[0],
        "review_count": len(messages),
        "score":        average,
        "sentiment":    sentiment,
        "score_distribution": {
            "positive": sum(1 for s in scores if s > 3.5),
            "neutral":  sum(1 for s in scores if 2.5 <= s <= 3.5),
            "negative": sum(1 for s in scores if s < 2.5),
        },
    })


def _background_reload_data():
    """Re-fetch Disqus data only; the pretrained model is not touched."""
    with _lock:
        _state["loading"] = True
        _state["error"]   = None
        _state["ready"]   = False
    try:
        df = load_disqus_data()
        with _lock:
            _state["df_reviews"] = df
            _state["ready"]      = True
            _state["progress"]   = f"Ready — {len(df)} reviews across {df['module_code'].nunique()} modules."
    except Exception as exc:
        with _lock:
            _state["error"]    = str(exc)
            _state["progress"] = f"Error: {exc}"
    finally:
        with _lock:
            _state["loading"] = False


@app.route("/api/reload", methods=["POST"])
def api_reload():
    """Re-fetch Disqus data without reloading the model."""
    with _lock:
        if _state["loading"]:
            return jsonify({"error": "Already loading."}), 409
    t = threading.Thread(target=_background_reload_data, daemon=True)
    t.start()
    return jsonify({"message": "Data reload started (model unchanged)."})



if __name__ == "__main__":
    start_init()                        
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
