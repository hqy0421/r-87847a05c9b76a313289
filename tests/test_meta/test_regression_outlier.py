import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import RegressionOutlierDetector


@parametrize_with_checks([RegressionOutlierDetector(LinearRegression(), column=0)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # Since `RegressionOutlierDetector` is an outlier detector (`OutlierMixin`), parametrize_with_checks feeds a
        # outlier dataset. However this is not how this componented is supposed to be used.
        "check_outliers_train",
        "check_fit2d_1feature",  # custom message
    }:
        pytest.skip()

    check(estimator)


def test_obvious_example():
    # generate random data for illustrative example
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 1))
    y = 1 + np.sum(X, axis=1).reshape(-1, 1) + np.random.normal(0, 0.2, (100, 1))
    for i in [20, 25, 50, 80]:
        y[i] += 10
    X = np.concatenate([X, y], axis=1)

    # fit and plot
    mod = RegressionOutlierDetector(LinearRegression(), column=1)
    preds = mod.fit(X).predict(X)
    for i in [20, 25, 50, 80]:
        assert preds[i] == -1


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame, pa.table])
def test_obvious_example_dataframe(frame_func):
    # generate random data for illustrative example
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 1 + x + np.random.normal(0, 0.2, 100)
    for i in [20, 25, 50, 80]:
        y[i] += 10
    X = frame_func({"x": x, "y": y})

    # fit and plot
    mod = RegressionOutlierDetector(LinearRegression(), column="y")
    preds = mod.fit(X).predict(X)
    for i in [20, 25, 50, 80]:
        assert preds[i] == -1


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame, pa.table])
def test_raises_error(frame_func):
    # generate random data for illustrative example
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 1 + x + np.random.normal(0, 0.2, 100)
    for i in [20, 25, 50, 80]:
        y[i] += 2
    X = frame_func({"x": x, "y": y})

    with pytest.raises(ValueError):
        mod = RegressionOutlierDetector(LogisticRegression(), column="y")
        mod.fit(X)


def test_named_target_is_excluded_from_regression():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    query = pd.DataFrame({"x": [5.0, 6.0], "target": [10.0, 100.0]})

    mod = RegressionOutlierDetector(LinearRegression(), column="target", lower=1, upper=1, method="sd")
    preds = mod.fit(train).predict(query)

    assert mod.estimator_.n_features_in_ == 1
    np.testing.assert_allclose(mod.estimator_.coef_, [2.0])
    assert mod.idx_ == 1
    np.testing.assert_array_equal(preds, [1, -1])


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame, pa.table])
@pytest.mark.parametrize("margin", [False, True])
def test_regression_pipeline_supported(frame_func, margin):
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 1 + x + np.random.normal(0, 0.2, 100)
    for i in [20, 25, 50, 80]:
        y[i] += 10
    X = frame_func({"x": x, "y": y})

    model = Pipeline([("scale", StandardScaler()), ("reg", LinearRegression())])
    mod = RegressionOutlierDetector(model, column="y", margin=margin)
    preds = mod.fit(X).predict(X)

    for i in [20, 25, 50, 80]:
        assert preds[i] == -1


def test_input_is_not_mutated():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 9.0]])
    X_before = X.copy()
    mod = RegressionOutlierDetector(LinearRegression(), column=1, margin=True).fit(X)
    mod.predict(X)
    mod.score_samples(X)
    np.testing.assert_array_equal(X, X_before)

    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    df_before = df.copy()
    mod = RegressionOutlierDetector(LinearRegression(), column="y", margin=True).fit(df)
    mod.predict(df)
    mod.score_samples(df)
    pd.testing.assert_frame_equal(df, df_before)


def _constant_predictor(value):
    """Regression model that always predicts `value`, to make the residual exactly controllable."""
    return DummyRegressor(strategy="constant", constant=value)


@pytest.mark.parametrize("method", ["sd", "absolute"])
def test_margin_mode_two_sided(method):
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = 3 + x + np.random.normal(0, 0.1, 200)
    data = np.column_stack([x, y])
    data[10, 1] += 10  # upper tail
    data[20, 1] -= 10  # lower tail

    kwargs = {"lower": 1, "upper": 1} if method == "absolute" else {}
    margin_mod = RegressionOutlierDetector(LinearRegression(), column=1, method=method, margin=True, **kwargs)
    signed_mod = RegressionOutlierDetector(LinearRegression(), column=1, method=method, **kwargs)

    margins = margin_mod.fit(data).score_samples(data)
    signed = signed_mod.fit(data).score_samples(data)

    # both tails are scored below zero...
    assert margins[10] < 0
    assert margins[20] < 0
    # ... while the signed residual gives them opposite signs.
    assert signed[10] > 0 > signed[20]
    # inliers stay non-negative
    inliers = np.ones(len(data), dtype=bool)
    inliers[[10, 20]] = False
    assert np.all(margins[inliers] >= 0)
    # hard labels are decided by the detector, not by the scoring mode
    np.testing.assert_array_equal(margin_mod.predict(data), signed_mod.predict(data))
    # `decision_function` uses the same scoring
    np.testing.assert_allclose(margin_mod.decision_function(data), margins)


def test_margin_boundary_is_inlier():
    data = np.array([[0.0, 4.0], [0.0, 5.0], [0.0, 6.0], [0.0, 7.0]])
    mod = RegressionOutlierDetector(
        _constant_predictor(5.0), column=1, method="absolute", lower=1, upper=1, margin=True
    )
    margins = mod.fit(data).score_samples(data)

    # residuals are -1, 0, 1, 2 -> distances to the band are 0, 1, 0, -1
    np.testing.assert_allclose(margins, [0.0, 1.0, 0.0, -1.0])
    assert np.all(mod.predict(data[:3]) == 1)
    assert mod.predict(data[3:])[0] == -1


def test_margin_zero_sd_does_not_divide_by_zero():
    constant = np.array([[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])
    mod = RegressionOutlierDetector(_constant_predictor(5.0), column=1, margin=True).fit(constant)
    assert mod.sd_ == 0

    scores = mod.score_samples(np.array([[0.0, 5.0], [0.0, 6.0], [0.0, 4.0]]))
    assert np.isfinite(scores).all()
    assert scores[0] == mod.upper  # residual 0 -> the whole band is left
    assert np.all(scores[1:] < 0)


def test_margin_relative_not_supported():
    data = np.array([[1.0, 2.0], [2.0, 4.0]])
    mod = RegressionOutlierDetector(LinearRegression(), column=1, method="relative", margin=True)
    with pytest.raises(ValueError, match="margin=True"):
        mod.fit(data)


def test_legacy_relative_scoring_is_unchanged():
    np.random.seed(42)
    x = np.random.normal(1, 1, 50)
    y = 2 + x + np.random.normal(0, 0.1, 50)
    data = np.column_stack([x, y])

    mod = RegressionOutlierDetector(LinearRegression(), column=1, method="relative").fit(data)
    expected = (y - mod.estimator_.predict(data[:, :1])) / y
    np.testing.assert_allclose(mod.score_samples(data), expected)
    np.testing.assert_allclose(mod.decision_function(data), expected)

    # `sd` and `absolute` keep returning the signed residual by default
    for method in ["sd", "absolute"]:
        mod = RegressionOutlierDetector(LinearRegression(), column=1, method=method).fit(data)
        np.testing.assert_allclose(mod.score_samples(data), y - mod.estimator_.predict(data[:, :1]))


def test_unknown_target_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    mod = RegressionOutlierDetector(LinearRegression(), column="missing")
    with pytest.raises(ValueError, match="does not match any column"):
        mod.fit(df)

    mod = RegressionOutlierDetector(LinearRegression(), column="a")
    with pytest.raises(ValueError, match="not a valid target index"):
        mod.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_bad_schema_raises():
    mod = RegressionOutlierDetector(LinearRegression(), column=0)
    with pytest.raises(ValueError, match="Expected 2D array"):
        mod.fit(np.array([1.0, 2.0, 3.0]))


def test_predict_preserves_row_order():
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = x + np.random.normal(0, 0.2, 100)
    y[[7, 33, 91]] += 20
    data = np.column_stack([x, y])

    mod = RegressionOutlierDetector(LinearRegression(), column=1, margin=True).fit(data)
    preds = mod.predict(data)

    # permuting the input permutes the output identically
    perm = np.random.permutation(len(data))
    np.testing.assert_array_equal(mod.predict(data[perm]), preds[perm])


@pytest.mark.parametrize("method", ["sd", "absolute"])
@pytest.mark.parametrize("frame_func", [np.asarray, pd.DataFrame])
def test_predict_agrees_with_margin_scores(method, frame_func):
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = 1 + x + np.random.normal(0, 0.15, 200)
    y[[5, 6, 7]] += [8, -8, 3]
    data = frame_func(np.column_stack([x, y]))

    mod = RegressionOutlierDetector(LinearRegression(), column=1, method=method, margin=True)
    mod.fit(data)
    scores = mod.score_samples(data)
    preds = mod.predict(data)

    # inliers have a non-negative margin, outliers a negative one
    assert np.array_equal(preds == -1, scores < 0)
    assert np.array_equal(preds == 1, scores >= 0)


def test_to_x_y_backwards_compatible():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mod = RegressionOutlierDetector(LinearRegression(), column=1).fit(data)
    X_to_use, y = mod.to_x_y(data)
    np.testing.assert_array_equal(X_to_use, data[:, [0, 2]])
    np.testing.assert_array_equal(y, data[:, 1])


@pytest.mark.parametrize(
    ("frame_func", "column"),
    [(np.asarray, 0), (pd.DataFrame, "target"), (pl.DataFrame, "target")],
)
def test_target_only_schema_raises(frame_func, column):
    data = frame_func({"target": [1.0, 2.0]}) if frame_func is not np.asarray else np.array([[1.0], [2.0]])
    mod = RegressionOutlierDetector(DummyRegressor(), column=column)
    with pytest.raises(ValueError, match="only contains the target column"):
        mod.fit(data)
