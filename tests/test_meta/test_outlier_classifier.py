import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn_compat.utils.estimator_checks import parametrize_with_checks

from sklego.meta import OutlierClassifier, RegressionOutlierDetector
from sklego.mixture import GMMOutlierDetector
from tests.conftest import GAUSSIAN_MIXTURE_ARRAY_API_REASON, expect_array_api_failure


@parametrize_with_checks(
    [OutlierClassifier(GMMOutlierDetector(threshold=0.1, method="quantile"))],
    expected_failed_checks=expect_array_api_failure(GAUSSIAN_MIXTURE_ARRAY_API_REASON),
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # Since `OutlierClassifier` is a classifier (`ClassifierMixin`), parametrize_with_checks feeds a classification
        # dataset. However this is not how this classifier is supposed to be used.
        "check_classifiers_train",
        "check_classifiers_classes",
        # Similarly, the original dataset could also be regression task depending on the outlier detection algo
        "check_classifiers_regression_target",
    }:
        pytest.skip()

    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.random.normal(0, 1, (2000, 2))


@pytest.mark.parametrize("outlier_model", [GMMOutlierDetector(), OneClassSVM(nu=0.05), IsolationForest()])
def test_obvious_usecase(dataset, outlier_model):
    outlier_clf = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(int)
    outlier_clf.fit(X, y)
    assert outlier_clf.predict([[10, 10]]) == np.array([1])
    assert outlier_clf.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(outlier_clf.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=3)
    np.testing.assert_allclose(outlier_clf.predict_proba([[10, 10]]), np.array([[0, 1]]), atol=0.2)
    assert isinstance(outlier_clf.score(X, y), float)


def test_raises_error(dataset):
    mod_quantile = LinearRegression()
    clf_quantile = OutlierClassifier(mod_quantile)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(int)
    with pytest.raises(ValueError):
        clf_quantile.fit(X, y)


def test_raises_error_no_decision_function(dataset):
    outlier_model = LocalOutlierFactor()
    clf_model = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(int)
    with pytest.raises(ValueError):
        clf_model.fit(X, y)


def _regression_frame():
    np.random.seed(42)
    n = 200
    x = np.random.normal(0, 1, n)
    target = 1 + x + np.random.normal(0, 0.15, n)
    target[5] += 8.0  # upper tail
    target[6] -= 8.0  # lower tail
    labels = np.zeros(n, dtype=int)
    labels[[5, 6]] = 1
    return pd.DataFrame({"x": x, "target": target}), labels


@pytest.mark.parametrize("method", ["sd", "absolute"])
def test_classifier_consumes_detector_margin(method):
    frame, labels = _regression_frame()
    kwargs = {"lower": 2, "upper": 2} if method == "absolute" else {}

    margin_clf = OutlierClassifier(
        RegressionOutlierDetector(LinearRegression(), column="target", method=method, margin=True, **kwargs)
    ).fit(frame, labels)
    signed_clf = OutlierClassifier(
        RegressionOutlierDetector(LinearRegression(), column="target", method=method, **kwargs)
    ).fit(frame, labels)

    margin_proba = margin_clf.predict_proba(frame)
    signed_proba = signed_clf.predict_proba(frame)

    # both tails are calibrated as outliers with the margin...
    assert margin_proba[5, 1] > margin_proba[0, 1]
    assert margin_proba[6, 1] > margin_proba[0, 1]
    # ... whereas the signed scoring only flags the upper tail.
    assert signed_proba[6, 1] < signed_proba[0, 1]
    # hard labels remain decided by the detector and identical for both scoring modes
    np.testing.assert_array_equal(margin_clf.predict(frame), signed_clf.predict(frame))
    np.testing.assert_array_equal(margin_clf.predict(frame)[[5, 6]], [1, 1])


def test_classifier_labels_do_not_replace_target():
    frame, labels = _regression_frame()
    clf = OutlierClassifier(RegressionOutlierDetector(LinearRegression(), column="target", margin=True))
    clf.fit(frame, labels)

    # the inner detector still predicts `target` from `x` only
    assert clf.estimator_.idx_ == 1
    assert clf.estimator_.estimator_.n_features_in_ == 1
    np.testing.assert_allclose(clf.estimator_.estimator_.coef_, [1.0], atol=0.1)
    assert clf.n_features_in_ == 2

    # flipping every label only changes the calibration, never the target column
    flipped = 1 - labels
    other = OutlierClassifier(RegressionOutlierDetector(LinearRegression(), column="target", margin=True))
    other.fit(frame, flipped)
    np.testing.assert_allclose(other.estimator_.estimator_.coef_, clf.estimator_.estimator_.coef_)
    np.testing.assert_array_equal(other.predict(frame), clf.predict(frame))


@pytest.mark.parametrize("frame_func", [pd.DataFrame, pl.DataFrame, pa.table])
def test_classifier_named_target_on_dataframes(frame_func):
    frame, labels = _regression_frame()
    frame = frame_func(frame)

    clf = OutlierClassifier(RegressionOutlierDetector(LinearRegression(), column="target", margin=True))
    clf.fit(frame, labels)
    preds = clf.predict(frame)
    proba = clf.predict_proba(frame)

    assert preds.shape == (len(labels),)
    assert proba.shape == (len(labels), 2)
    np.testing.assert_array_equal(preds[[5, 6]], [1, 1])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_requires_y():
    frame, _ = _regression_frame()
    clf = OutlierClassifier(RegressionOutlierDetector(LinearRegression(), column="target", margin=True))
    with pytest.raises(ValueError, match="requires y to be passed"):
        clf.fit(frame)


def test_classifier_rejects_non_finite_y():
    frame, labels = _regression_frame()
    clf = OutlierClassifier(RegressionOutlierDetector(LinearRegression(), column="target", margin=True))

    nan_labels = labels.astype(float)
    nan_labels[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        clf.fit(frame, nan_labels)

    inf_labels = labels.astype(float)
    inf_labels[3] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        clf.fit(frame, inf_labels)
