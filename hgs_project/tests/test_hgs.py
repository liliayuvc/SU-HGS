"""
test_hgs.py – Unit tests for the HGS implementation.

Run with:  pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.heuristics import HGS, Hypothesis, WILDCARD, NONE_SYM
from src.preprocessing import load_dataset, preprocess, train_test_split
from src.puzzle_utils import evaluate


# ================================================================= #
#  Hypothesis unit tests                                            #
# ================================================================= #

class TestHypothesis:
    def test_most_general_covers_anything(self):
        h = Hypothesis.most_general(3)
        assert h.covers(["a", "b", "c"])
        assert h.covers(["x", "y", "z"])

    def test_most_specific_covers_nothing(self):
        h = Hypothesis.most_specific(3)
        assert not h.covers(["a", "b", "c"])

    def test_concrete_hypothesis_covers_exact_match(self):
        h = Hypothesis(["Sunny", WILDCARD, "High"])
        assert h.covers(["Sunny", "anything", "High"])
        assert not h.covers(["Rainy", "anything", "High"])
        assert not h.covers(["Sunny", "anything", "Low"])

    def test_is_more_general_than(self):
        h_general  = Hypothesis.most_general(3)
        h_specific = Hypothesis(["Sunny", WILDCARD, "High"])
        assert h_general.is_more_general_than(h_specific)
        assert not h_specific.is_more_general_than(h_general)

    def test_wildcard_str_representation(self):
        h = Hypothesis.most_general(2)
        assert str(h) == "<?, ?>"

    def test_none_sym_str_representation(self):
        h = Hypothesis.most_specific(2)
        assert NONE_SYM in str(h)


# ================================================================= #
#  HGS algorithm tests                                              #
# ================================================================= #

class TestHGS:
    def _weather_data(self):
        """Classic Mitchell weather dataset."""
        X = [
            ["Sunny",  "Warm", "Normal", "Strong", "Warm",  "Same"],
            ["Sunny",  "Warm", "High",   "Strong", "Warm",  "Same"],
            ["Rainy",  "Cold", "High",   "Strong", "Warm",  "Change"],
            ["Sunny",  "Warm", "High",   "Strong", "Cool",  "Change"],
        ]
        y = [1, 1, 0, 1]
        return X, y

    def test_fit_does_not_raise(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)

    def test_g_set_not_empty_after_fit(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        assert len(model.g_set) > 0

    def test_predict_shape(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_positive_examples_covered(self):
        """All positive training examples should be classified positive."""
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        preds = model.predict(X)
        for xi, yi, pi in zip(X, y, preds):
            if yi == 1:
                assert pi == 1, f"Positive example {xi} not covered"

    def test_negative_example_not_in_training_covered_correctly(self):
        """Negative training examples should NOT be covered after training."""
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        neg_examples = [xi for xi, yi in zip(X, y) if yi == 0]
        preds = model.predict(neg_examples)
        for pred in preds:
            assert pred == 0

    def test_predict_proba_shape(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        proba = model.predict_proba(X)
        for row in proba:
            assert abs(row[0] + row[1] - 1.0) < 1e-9

    def test_history_length(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        assert len(model.history_) == len(X)

    def test_only_positive_examples(self):
        X = [["A", "B"], ["C", "D"], ["A", "D"]]
        y = [1, 1, 1]
        model = HGS()
        model.fit(X, y)
        # Version space should still be non-empty
        assert not model.is_version_space_empty

    def test_only_negative_examples(self):
        X = [["A", "B"], ["C", "D"]]
        y = [0, 0]
        model = HGS()
        model.fit(X, y)
        # With concrete-value specialisation, G-set contains hypotheses
        # that exclude all negative examples seen so far
        for h in model.g_set:
            for example in X:
                assert not h.covers(example), \
                    f"Hypothesis {h} incorrectly covers negative example {example}"

    def test_feature_names_stored(self):
        X, y = self._weather_data()
        model = HGS(feature_names=["Sky", "Temp", "Hum", "Wind", "Water", "Forecast"])
        model.fit(X, y)
        assert model.feature_names is not None
        assert len(model.feature_names) == 6

    def test_summary_string(self):
        X, y = self._weather_data()
        model = HGS()
        model.fit(X, y)
        s = model.summary()
        assert "HGS Summary" in s
        assert "G-set size" in s


# ================================================================= #
#  Preprocessing tests                                              #
# ================================================================= #

class TestPreprocessing:
    def test_load_weather(self):
        X, y, feature_names = load_dataset("weather")
        assert len(X) > 0
        assert len(y) == len(X)
        assert len(feature_names) == len(X[0])

    def test_load_animals(self):
        X, y, feature_names = load_dataset("animals")
        assert len(X) > 0

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            load_dataset("unknown_dataset_xyz")

    def test_preprocess_returns_same_length(self):
        X, y, _ = load_dataset("weather")
        X2, y2 = preprocess(X, y)
        assert len(X2) == len(X)
        assert len(y2) == len(y)

    def test_train_test_split_sizes(self):
        X, y, _ = load_dataset("animals")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_seed=0)
        total = len(X_tr) + len(X_te)
        assert total == len(X)


# ================================================================= #
#  Evaluation tests                                                 #
# ================================================================= #

class TestEvaluation:
    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        m = evaluate(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = [1, 1, 1]
        y_pred = [0, 0, 0]
        m = evaluate(y_true, y_pred)
        assert m["accuracy"] == 0.0
        assert m["recall"] == 0.0

    def test_metrics_keys_present(self):
        m = evaluate([1, 0], [1, 0])
        for key in ["accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]:
            assert key in m

    def test_f1_harmonic_mean(self):
        # P = 1.0, R = 0.5 → F1 = 2/3
        y_true = [1, 1, 0]
        y_pred = [1, 0, 0]
        m = evaluate(y_true, y_pred)
        assert abs(m["f1"] - 2/3) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
