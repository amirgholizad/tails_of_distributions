import functions
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("ticker, period1, period2, interval, expected",
                         [("TSLA", "2023-11-24", "2023-11-28", "1d", 236.08)])
def test_yahoo_data(ticker, period1, period2, interval, expected):
    observed = functions.yahoodata(ticker, period1, period2, interval).getdata()
    assert round(observed["Close"][-1], 2) == expected


@pytest.mark.parametrize("historical_data, expected",
                         [(pd.DataFrame([1,2,3,4,5,6,7,8,9,10]), np.array([0.69314718, 0.40546511,
                                                    0.28768207, 0.22314355,
                                                    0.18232156, 0.15415068,
                                                    0.13353139, 0.11778304, 0.10536052]))])
def test_log_return(historical_data, expected):
    observed = functions.log_returns(historical_data)
    assert observed.dropna().to_numpy().squeeze() == expected