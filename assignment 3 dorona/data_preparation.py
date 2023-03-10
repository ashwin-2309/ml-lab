import numpy as np
import pandas as pd


def prepare_data(start_year, end_year, pct_train):
    years = np.arange(start_year, end_year + 1)
    time = years - 1900
    d = 100 * (2.4) ** time + np.random.normal(0, 1, len(time))
    d_0 = np.array([100] * len(time))

    log_d = np.log(d)

    train_size = int(pct_train * len(time))
    train_time = time[:train_size]
    test_time = time[train_size:]
    train_log_d = log_d[:train_size]
    test_log_d = log_d[train_size:]

    return train_time, test_time, train_log_d, test_log_d
