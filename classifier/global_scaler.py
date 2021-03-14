# 9 Feb 2021
import numpy as np
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale
from sklearn.utils.validation import FLOAT_DTYPES

class GlobalMinMaxScaler(MinMaxScaler):
    """Transform features by scaling all features together to a given range.

    Derives from MinMaxScaler; that scales each feature individually
    """

#    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
#        self.feature_range = feature_range
#        self.copy = copy
#        self.clip = clip

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does not support sparse input. "
                            "Consider using MaxAbsScaler instead.")

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        # Here's the difference - take min and max over ALL features
        data_min = np.nanmin(X)
        data_max = np.nanmax(X)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self
