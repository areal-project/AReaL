# SPDX-License-Identifier: Apache-2.0

# The following code is adapted with minor modifications from
# https://github.com/Whisper-6/DynamicTreeAttn/blob/main/tree_time_model.py.
import numpy as np
from scipy.optimize import nnls


class TreeTimeModel:
    MIN_N_DATA_POINTS = 16
    MAX_N_DATA_POINTS = 1024

    def __init__(self):
        # T = c_0 * n_leaf_sequences + c_1 * n_tree_tokens + c_2 * n_f1_tokens + c_3 * sum_prefix_len + c_4 * sum_depth
        self.coeffs = None
        self.data = []

    def fit(self):
        X, Y = [], []
        for stats in self.data:
            # X.append([0, stats["n_tree_tokens"], 0, 0, 0])
            X.append(
                [
                    stats["n_leaf_sequences"],
                    stats["n_tree_tokens"],
                    stats.get("n_f1_tokens", 0),
                    stats["sum_prefix_len"],
                    stats["sum_depth"],
                ]
            )
            Y.append(stats["time"])

        X, Y = np.array(X), np.array(Y)
        self.coeffs, _ = nnls(X, Y)

        T_pred = X @ self.coeffs
        mse = np.mean((T_pred - Y) ** 2)
        return mse

    def add_data(self, data):
        self.data.extend(data)
        if len(self.data) > self.MAX_N_DATA_POINTS:
            self.data = self.data[-self.MAX_N_DATA_POINTS :]
        if len(self.data) >= self.MIN_N_DATA_POINTS:
            self.fit()

    def pred(self, stats):
        if self.coeffs is None:
            return stats["n_tree_tokens"]
        return (
            self.coeffs[0] * stats["n_leaf_sequences"]
            + self.coeffs[1] * stats["n_tree_tokens"]
            + self.coeffs[2] * stats.get("n_f1_tokens", 0)
            + self.coeffs[3] * stats["sum_prefix_len"]
            + self.coeffs[4] * stats["sum_depth"]
        )
