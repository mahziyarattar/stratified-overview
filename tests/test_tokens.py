from stratified_overview.tokens import DEFAULT_SCALES, TiktokenCounter, TokenCalibration


def test_scale_multiplies_raw_count():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    plain, scaled = TiktokenCounter(scale=1.0), TiktokenCounter(scale=1.5)
    assert scaled.count(text) >= plain.count(text) * 1.5 - 1


def test_truncate_fits_scaled_budget():
    counter = TiktokenCounter(scale=1.63)
    text = "word " * 2_000
    assert counter.count(counter.truncate(text, 300)) <= 300 + 2


def test_calibration_defaults_then_learns(tmp_path):
    path = tmp_path / "calibration.json"
    calibration = TokenCalibration(path)
    assert calibration.scale_for("anthropic/claude-haiku-5.5") == DEFAULT_SCALES["anthropic/claude-haiku-5.5"]
    assert calibration.scale_for("unknown-model") == 1.0

    assert calibration.record("unknown-model", raw_tokens=1_000, billed_tokens=1_250) == 1.25
    assert TokenCalibration(path).scale_for("unknown-model") == 1.25
