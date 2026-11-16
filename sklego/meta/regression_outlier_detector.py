import operator

import narwhals.stable.v1 as nw
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, OutlierMixin, is_regressor
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

_MAX_FLOAT = np.finfo(np.float64).max


class RegressionOutlierDetector(OutlierMixin, BaseEstimator):
    """Morphs a regression estimator into one that can detect outliers. We will try to predict `column` in X.

    The target column is never passed to the regression model as a feature, whether it is referred to by name
    (dataframe input) or by position (array input). It is only used to compare the model's prediction with the value
    that was actually observed.

    Parameters
    ----------
    model : scikit-learn compatible regression model
        A regression model that will be used for prediction. Pipelines that end in a regressor are supported.
    column : int | str
        This should be:
            - The index of the target column to predict in the input data, when the input is an array.
            - The name of the target column to predict in the input data, when the input is a dataframe.
        A `ValueError` is raised when the target column cannot be found.
    lower : float, default=2.0
        Lower threshold for outlier detection. The method used for detection depends on the `method` parameter.
    upper : float, default=2.0
        Upper threshold for outlier detection. The method used for detection depends on the `method` parameter.
    method : Literal["sd", "relative", "absolute"], default="sd"
        The method to use for outlier detection.

        - `"sd"` uses standard deviation
        - `"relative"` uses relative difference
        - `"absolute"` uses absolute difference
    margin : bool, default=False
        Whether `score_samples` and `decision_function` return a two-sided margin instead of the signed residual.

        - With `margin=False` the signed residual `y - y_pred` is returned for `method="sd"` and `method="absolute"`,
          and `(y - y_pred) / y` for `method="relative"`. Samples in the lower and the upper tail then receive scores
          with opposite signs.
        - With `margin=True` the score is the smallest signed distance from the residual to the upper and the lower
          threshold: `min(upper - z, z + lower)`, where `z = (y - y_pred) / sd_` for `method="sd"` and
          `z = y - y_pred` for `method="absolute"`. The score is non-negative inside the band (the boundaries
          included) and negative outside of it, so both tails score below zero.
        - If `sd_ == 0` the normalization saturates at the largest finite float instead of dividing by zero, so scores
          remain finite.
        - `margin=True` is not supported for `method="relative"`.

    Attributes
    ----------
    estimator_ : scikit-learn compatible regression model
        The fitted underlying regression model.
    sd_ : float
        The standard deviation of the differences between true and predicted values.
    idx_ : int
        The (positional) index of the target column in the input data.
    offset_ : float
        Offset subtracted by `decision_function`, always `0`.
    n_features_in_ : int
        Number of columns (including the target column) seen during `fit`.
    feature_names_in_ : np.ndarray of shape (n_features_in_,)
        Column names seen during `fit`, only defined when the input was a dataframe.

    Notes
    -----
    Native cross-dataframe support is achieved using
    [Narwhals](https://narwhals-dev.github.io/narwhals/){:target="_blank"}.
    Supported dataframes are:

    - pandas
    - Polars (eager)
    - Modin

    See [Narwhals docs](https://narwhals-dev.github.io/narwhals/extending/){:target="_blank"} for an up-to-date list
    (and to learn how you can add your dataframe library to it!), though note that only those
    supported by [sklearn.utils.check_X_y](https://scikit-learn.org/stable/modules/generated/sklearn.utils.check_X_y.html)
    will work with this class.

    The input data is never modified and the rows of `X` are never reordered, so `predict` and `score_samples`
    return one value per input row, in the original order.

    Example
    -------

    ```py
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklego.meta import RegressionOutlierDetector

    np.random.seed(0)
    n1, n2 = 30, 5
    X = np.random.normal(0, 1, (n1+n2, 2))
    y = np.concatenate([np.random.normal(0, 0.5, (n1, 1)), np.random.normal(5, 1, (n2,1))], axis=0)
    data = np.concatenate([X, y], axis=1)

    col = 2

    # Initialize the outlier detector - outliers are points predicted more that +/-1 sd from mean
    linear_regressor = LinearRegression()
    outlier_detector = RegressionOutlierDetector(linear_regressor, col, lower=1, upper=1)

    outlier_detector.fit(data)
    preds = outlier_detector.predict(data)
    scores = outlier_detector.decision_function(data)

    print(preds)
    ### The last 5 points are outliers, as expected
    ### [ 1  1  1  1  1  1  1  1  1  1 -1  1  1  1  1  1  1  1  1  1 -1  1  1  1 1  1  1  1  1  1 -1 -1 -1 -1 -1]
    ```
    """

    _ALLOWED_METHODS = ("sd", "relative", "absolute")
    _MARGIN_METHODS = ("sd", "absolute")

    _required_parameters = ["model", "column"]

    def __init__(self, model, column, lower=2, upper=2, method="sd", margin=False):
        self.model = model
        self.column = column
        self.lower = lower
        self.upper = upper
        self.method = method
        self.margin = margin

    def _is_regression_model(self):
        """Check if the underlying model is a regression model."""
        try:
            return is_regressor(self.model)
        except AttributeError:
            # Not a scikit-learn estimator at all.
            return False

    def _validate_params(self):
        """Validate the scoring parameters. Raises a `ValueError` when they are inconsistent."""
        if self.method not in self._ALLOWED_METHODS:
            raise ValueError(f"`method` must be in {self._ALLOWED_METHODS} got: {self.method}")
        if self.margin and self.method not in self._MARGIN_METHODS:
            raise ValueError(
                f"`margin=True` is only supported for method in {self._MARGIN_METHODS} got: {self.method}. "
                "Use `margin=False` to keep the legacy relative deviation."
            )

    @staticmethod
    def _is_dataframe(X):
        return nw.dependencies.is_into_dataframe(X)

    def _target_name(self, X):
        """Return the name of the target column of the dataframe `X`.

        Raises a `ValueError` when the target column is not part of `X`.
        """
        frame = nw.from_native(X, eager_only=True, strict=False)
        columns = list(frame.columns)
        for name in columns:
            if name == self.column:
                return name
        raise ValueError(
            f"`column`={self.column!r} does not match any column in X. Available columns are {columns}. When X is a "
            "dataframe, `column` must be the name of the target column."
        )

    def _target_index(self, n_features):
        """Return the positional index of the target column for an array-like `X`."""
        try:
            idx = operator.index(self.column)
        except TypeError:
            idx = None
        # `bool` is an `int` subclass, but `column=True` is a mistake rather than a valid index.
        if idx is None or self.column is True or self.column is False:
            raise ValueError(
                f"`column`={self.column!r} is not a valid target index for an array-like X. When X is an array, "
                "`column` must be an integer index."
            )
        if idx < 0:
            idx += n_features
        if not 0 <= idx < n_features:
            raise ValueError(f"`column`={self.column!r} is out of range for an X with {n_features} column(s).")
        return idx

    def _split(self, X, *, reset):
        """Split `X` into the features to predict with and the target column.

        The target column is excluded from the features by its positional index, so it cannot leak into the model when
        a dataframe column name was used. The row order of `X` is preserved.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Data to split.
        reset : bool
            Whether this is the initial `fit` call (resolving and storing `idx_`).

        Returns
        -------
        X_to_use : array-like of shape (n_samples, n_features - 1)
            Data to use for prediction.
        y : np.ndarray of shape (n_samples,)
            The target column.
        """
        if self._is_dataframe(X):
            X_checked = validate_data(self, X=X, reset=reset, skip_check_array=True)
            frame = nw.from_native(X_checked, eager_only=True, strict=False)
            name = self._target_name(frame)
            idx = list(frame.columns).index(name)
            if len(frame.columns) == 1:
                raise ValueError(
                    "X only contains the target column, so there are no features left to predict it from. Pass an X "
                    "that contains at least one feature besides the target column."
                )
            X_to_use = frame.drop(name).to_native()
            y = frame.get_column(name).to_numpy()
        else:
            X_checked = validate_data(self, X=X, reset=reset)
            n_features = X_checked.shape[1]
            idx = self._target_index(n_features) if reset else self.idx_
            if not 0 <= idx < n_features:
                raise ValueError(
                    f"`column`={self.column!r} (resolved to index {idx}) is out of range for an X with "
                    f"{n_features} column(s)."
                )
            if n_features == 1:
                raise ValueError(
                    "X only contains the target column, so there are no features left to predict it from. Pass an X "
                    "that contains at least one feature besides the target column."
                )
            X_to_use = np.delete(X_checked, idx, axis=1)
            y = X_checked[:, idx]

        if reset:
            self.idx_ = idx
        return X_to_use, y

    def to_x_y(self, X):
        """Split `X` into the features to predict with and the target column.

        `y` is the column we want to predict (specified in the `column` parameter) and `X_to_use` is the rest of the
        data. This is a convenience wrapper around the internal split logic and is kept for backwards compatibility.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Data to split.

        Returns
        -------
        X_to_use : array-like of shape (n_samples, n_features - 1)
            Data to use for prediction.
        y : np.ndarray of shape (n_samples,)
            The target column.
        """
        check_is_fitted(self, ["idx_"])
        return self._split(X, reset=False)

    def _handle_thresholds(self, y_true, y_pred):
        """Compute if a sample is an outlier based on the `method` parameter."""
        difference = y_true - y_pred
        results = np.ones(difference.shape, dtype=int)
        if self.method not in self._ALLOWED_METHODS:
            raise ValueError(f"`method` must be in {self._ALLOWED_METHODS} got: {self.method}")
        if self.method == "sd":
            lower_limit_hit = -self.lower * self.sd_ > difference
            upper_limit_hit = self.upper * self.sd_ < difference
        if self.method == "relative":
            lower_limit_hit = -self.lower > difference / y_true
            upper_limit_hit = self.upper < difference / y_true
        if self.method == "absolute":
            lower_limit_hit = -self.lower > difference
            upper_limit_hit = self.upper < difference
        results[lower_limit_hit] = -1
        results[upper_limit_hit] = -1
        return results

    def _margin_scores(self, difference):
        """Compute the smallest signed distance from the residual to the lower and upper thresholds.

        The score is non-negative inside the band and negative outside of it, so that both tails are scored below
        zero. When `sd_` is zero, the normalized residual saturates at the largest finite float rather than dividing
        by zero.
        """
        difference = np.asarray(difference, dtype=np.float64)
        if self.method == "sd":
            if self.sd_ == 0:
                normalized = np.zeros_like(difference)
                outside = difference != 0
                normalized[outside] = np.copysign(_MAX_FLOAT, difference[outside])
            else:
                normalized = difference / self.sd_
        else:
            normalized = difference
        margin = np.minimum(self.upper - normalized, normalized + self.lower)
        # `upper - normalized` or `normalized + lower` can overflow for extreme residuals, keep the score finite.
        return np.clip(margin, -_MAX_FLOAT, _MAX_FLOAT)

    def fit(self, X, y=None):
        """Fit the underlying model on `X_to_use` and `y` where:

        - `y` is the column we want to predict (`X[:, self.column]` or `X[self.column]`)
        - `X_to_use` is the rest of the data.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : RegressionOutlierDetector
            The fitted estimator.

        Raises
        ------
        ValueError
            - If the `model` is not a regression estimator.
            - If the target column cannot be found in `X`.
            - If `method` is not supported, or `margin=True` is combined with `method="relative"`.
        """
        if not self._is_regression_model():
            raise ValueError(f"Passed model must be a regression model, got {type(self.model).__name__}!")
        self._validate_params()

        X, y = self._split(X, reset=True)
        self.estimator_ = clone(self.model).fit(X, y)
        self.sd_ = np.std(self.estimator_.predict(X) - y)
        self.offset_ = 0

        return self

    def predict(self, X, y=None):
        """Predict which samples of `X` are outliers using the underlying estimator and given `method`.

        The hard labels are always decided by the detector itself, regardless of `margin`.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            The data to predict.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values. 1 for inliers, -1 for outliers.
        """
        check_is_fitted(self, ["estimator_", "sd_", "idx_"])
        X, y_true = self._split(X, reset=False)
        preds = self.estimator_.predict(X)
        return self._handle_thresholds(y_true, preds)

    def score_samples(self, X, y=None):
        """Calculate the outlier scores for the given data.

        By default the signed residual is returned, see the `margin` parameter for the two-sided margin.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Data for which outlier scores are calculated.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The outlier scores for the input data.

        Raises
        ------
        ValueError
            If `method` is not one of "sd", "relative", or "absolute", or if `margin=True` is combined with
            `method="relative"`.
        """
        check_is_fitted(self, ["estimator_", "sd_", "idx_"])
        self._validate_params()

        X, y_true = self._split(X, reset=False)
        y_pred = self.estimator_.predict(X)
        difference = y_true - y_pred

        if self.margin:
            return self._margin_scores(difference)
        if self.method == "relative":
            return difference / y_true
        return difference

    def decision_function(self, X):
        """Calculate the outlier scores for the given data, see `score_samples`.

        Parameters
        ----------
        X : array-like or dataframe of shape (n_samples, n_features)
            Data for which outlier scores are calculated.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The outlier scores for the input data.
        """
        return self.score_samples(X) - self.offset_
