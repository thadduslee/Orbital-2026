import pandas as pd
from app import (
    clean_review_text,
    module_code_from_title,
    sentiment_from_average,
    get_academic_period,
    period_sort_key,
    build_semester_sentiment,
)

def test_clean_review_text_strips_boilerplate_and_links():
    raw = "Module review by John Tan: Taken in AY2023/2024 Sem 1. Great mod! http://example.com"
    result = clean_review_text(raw)
    assert "http" not in result
    assert "Taken in AY" not in result
    assert "Great mod" in result

def test_clean_review_text_handles_empty_input():
    assert clean_review_text("") == ""
    assert clean_review_text(None) == ""

def test_module_code_from_title_valid_code():
    assert module_code_from_title("CS2103T Software Engineering") == "CS2103T"

def test_module_code_from_title_rejects_invalid_prefix():
    assert module_code_from_title("Random discussion thread") == "UNKNOWN"

def test_sentiment_from_average_boundaries():
    assert sentiment_from_average(4.0) == "positive"
    assert sentiment_from_average(2.0) == "negative"
    assert sentiment_from_average(3.5) == "neutral"   # boundary not > 3.5
    assert sentiment_from_average(2.5) == "neutral"   # boundary  not < 2.5
    assert sentiment_from_average(3.0) == "neutral"

def test_get_academic_period_sem1_and_sem2_and_special_term():
    assert get_academic_period("2024-09-15") == "AY2024/2025 Sem 1"
    assert get_academic_period("2024-03-01") == "AY2023/2024 Sem 2"
    assert get_academic_period("2024-06-15") == "AY2023/2024 Special Term"

def test_period_sort_key_orders_semesters_chronologically():
    periods = ["AY2024/2025 Sem 2", "AY2023/2024 Sem 1", "AY2024/2025 Sem 1"]
    assert sorted(periods, key=period_sort_key) == [
        "AY2023/2024 Sem 1", "AY2024/2025 Sem 1", "AY2024/2025 Sem 2",
    ]

def test_build_semester_sentiment_flags_low_sample_periods_unreliable():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-09-01", "2024-09-02"], utc=True),
        "module_code": ["CS2103T", "CS2103T"],
    })
    result = build_semester_sentiment(df, scores=[4.0, 4.5])
    assert result[0]["reliable"] is False  # only 2 reviews when MIN_SAMPLES = 5
    assert result[0]["period"] == "AY2024/2025 Sem 1"