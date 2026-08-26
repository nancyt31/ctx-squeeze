from ctx_squeeze.compactor import STRATEGIES, squeeze
from ctx_squeeze.tokens import estimate_tokens


def _doc(*paragraphs):
    return "\n\n".join(paragraphs)


def test_squeeze_keeps_everything_when_budget_covers_total():
    text = _doc("first paragraph about the release", "second paragraph about the rollout")
    result = squeeze(text, budget=estimate_tokens(text))
    assert result.segments_out == result.segments_in
    assert result.text == text


def test_squeeze_drops_a_segment_and_still_fits_the_budget():
    text = _doc(
        "first paragraph about the release process",
        "second paragraph about the database migration",
        "third paragraph about the rollout schedule",
    )
    result = squeeze(text, budget=30)
    assert result.segments_out < result.segments_in
    assert result.final_tokens <= 30


def test_squeeze_reports_original_and_final_tokens():
    text = _doc("short paragraph")
    result = squeeze(text, budget=1000)
    assert result.original_tokens == estimate_tokens(text)
    assert result.final_tokens == estimate_tokens(result.text)


def test_squeeze_inserts_elision_marker_for_dropped_segments():
    text = _doc(
        "first paragraph about the release process here",
        "second paragraph about something else entirely",
        "third paragraph about the rollout schedule today",
    )
    result = squeeze(text, budget=1, strategy="score")
    assert result.text == "" or "elided" in result.text


def test_squeeze_omits_marker_when_marker_is_false():
    text = _doc(
        "first paragraph about the release process here",
        "second paragraph about something else entirely",
    )
    result = squeeze(text, budget=1, strategy="score", marker=False)
    assert "elided" not in result.text


def test_squeeze_dedupe_then_score_drops_duplicate_before_scoring():
    repeated = "The build failed after the runner image was bumped last night here."
    text = _doc(
        repeated,
        "Something unrelated about the release notes for this quarter today.",
        repeated,
    )
    result = squeeze(text, budget=1000, strategy="dedupe,score")
    assert result.segments_out == 2
    assert any("dedupe dropped" in note for note in result.notes)


def test_squeeze_head_tail_keeps_a_prefix_and_a_suffix():
    text = _doc(
        "alpha paragraph about the start of things",
        "beta paragraph in the middle nobody needs",
        "gamma paragraph in the middle nobody needs either",
        "delta paragraph about the end of things",
    )
    result = squeeze(text, budget=20, strategy="head-tail", head_ratio=0.5)
    assert "alpha" in result.text
    assert "delta" in result.text
    assert "beta" not in result.text


def test_squeeze_unknown_strategy_raises():
    try:
        squeeze("some text", budget=100, strategy="not-a-real-stage")
    except ValueError as error:
        assert "not-a-real-stage" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_squeeze_empty_strategy_raises():
    try:
        squeeze("some text", budget=100, strategy="  ,  ")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_squeeze_of_empty_text_is_empty():
    result = squeeze("", budget=100)
    assert result.text == ""
    assert result.segments_in == 0
    assert result.segments_out == 0


def test_squeeze_zero_budget_returns_empty_text_with_marker():
    text = _doc("only paragraph in the document")
    result = squeeze(text, budget=0)
    assert result.text == "[1 segments elided]"


def test_strategies_exposes_every_documented_stage_name():
    assert set(STRATEGIES) == {"dedupe", "score", "head-tail"}
