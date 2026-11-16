import narwhals.stable.v1 as nw
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn_compat.utils.validation import validate_data

from sklego.base import OutlierModel


class OutlierClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Morphs an outlier detection model into a classifier.

    When an outlier is detected it will output 1 and 0 otherwise. This way you can use familiar metrics again and this
    allows you to consider outlier models as a fraud detector.

    The labels `y` passed to `fit` are only used to calibrate the probabilities returned by `predict_proba`. They never
    replace, relabel or otherwise influence the target that the wrapped outlier model uses internally. The hard labels
    returned by `predict` are always the ones decided by the wrapped outlier model.

    Both the calibration during `fit` and the probabilities returned by `predict_proba` are computed from the wrapped
    model's `decision_function`. For a `RegressionOutlierDetector` with `margin=True` this means both consume the
    two-sided margin: samples that deviate too far below *or* above the regression are scored consistently.

    Parameters
    ----------
    model : scikit-learn compatible outlier detection model
        An outlier detection model that will be used for prediction.

    Attributes
    ----------
    estimator_ : scikit-learn compatible outlier detection model
        The fitted underlying outlier detection model.
    classes_ : array-like of shape (2,)
        Classes used for prediction (0 or 1)
    n_features_in_ : int
        Number of features seen during `fit`.
    feature_names_in_ : np.ndarray of shape (n_features_in_,)
        Feature names seen during `fit`, only defined when the input was a dataframe.

    Example
    -------
    ```py
    from sklearn.ensemble import IsolationForest
    from sklego.meta.outlier_classifier import OutlierClassifier

    X = [[0], [0.5], [-1], [99]]
    y = [0, 0, 0, 1]

    isolation_forest = IsolationForest()

    outlier_clf = OutlierClassifier(isolation_forest)
    _ = outlier_clf.fit(X, y)

    preds = outlier_clf.predict([[100], [-0.5], [0.5], [1]])
    # array[1. 0. 0. 0.]

    proba_preds = outlier_clf.predict_proba([[100], [-0.5], [0.5], [1]])
    # [[0.34946567 0.65053433]
    #  [0.79707913 0.20292087]
    #  [0.80275406 0.19724594]
    #  [0.80275406 0.19724594]]
    ```
    """

    _required_parameters = ["model"]

    def __init__(self, model):
        self.model = model

    def _is_outlier_model(self):
        """Check if the underlying model is an outlier detection model."""
        return isinstance(self.model, OutlierModel)

    def _validate_input(self, X, y=None, *, reset):
        """Validate `X` (and `y`), keeping dataframes intact when they are passed.

        Dataframes are not converted to arrays, so models that select their target by column name keep working. For
        array-like input the regular scikit-learn validation (including the finite-value checks) is applied.
        """
        if nw.dependencies.is_into_dataframe(X):
            X = validate_data(self, X=X, reset=reset, skip_check_array=True)
            if y is not None:
                y = column_or_1d(y, warn=True)
                y = check_array(y, ensure_2d=False, ensure_all_finite=True, dtype=None, input_name="y")
            return X, y
        if y is None:
            return validate_data(self, X=X, reset=reset), None
        return validate_data(self, X=X, y=y, reset=reset)

    def _outlier_scores(self, X):
        """Return the scores that feed the probability calibration and `predict_proba`."""
        return self.estimator_.decision_function(X)

    def fit(self, X, y=None):
        """Fit the underlying estimator to the training data `X` and `y`.

        The labels `y` are only used to calibrate `predict_proba`. The wrapped model is fitted on `X` as-is, so a
        `RegressionOutlierDetector` keeps using its own target column and, when `margin=True`, its margin scores.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values, only used for the probability calibration.

        Returns
        -------
        self : OutlierClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            - If the underlying model is not an outlier detection model.
            - If the underlying model does not have a `decision_function` method.
            - If `y` is not passed.
        """
        if not self._is_outlier_model():
            raise ValueError("Passed model does not detect outliers!")
        if not hasattr(self.model, "decision_function"):
            raise ValueError(
                f"Passed model {self.model} does not have a `decision_function` "
                f"method. This is required for `predict_proba` estimation."
            )
        if y is None:
            raise ValueError(f"This {type(self).__name__} estimator requires y to be passed, but the target y is None.")
        # Keep dataframes intact so that models which select their target by column name keep working. The wrapped
        # model validates the data it consumes; `y` is only needed for the calibration below.
        X, y = self._validate_input(X, y, reset=True)

        self.estimator_ = clone(self.model).fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        self.classes_ = np.array([0, 1])

        # fit sigmoid function for `predict_proba`
        self._predict_proba_sigmoid = _SigmoidCalibration().fit(self._outlier_scores(X), y)

        return self

    def predict(self, X):
        """Predict new data.

        The labels are decided by the wrapped outlier model, regardless of how its probabilities are calibrated.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values. 0 for inliers, 1 for outliers.
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        X, _ = self._validate_input(X, reset=False)
        preds = self.estimator_.predict(X)
        result = (preds == -1).astype(int)
        return result

    def predict_proba(self, X):
        """Predict probability estimates for new data.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            The predicted probabilities, `[P(inlier), P(outlier)]`.
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        X, _ = self._validate_input(X, reset=False)
        decision_function_scores = self._outlier_scores(X)
        probabilities = self._predict_proba_sigmoid.predict(decision_function_scores).reshape(-1, 1)
        complement = np.ones_like(probabilities) - probabilities
        return np.hstack((complement, probabilities))
