"""Tools for looking at *where* a direction classifier goes wrong.

All functions work on plain pandas objects. They are diagnostics: they help
describe errors and form hypotheses, they are not meant to tune the model on
the same data they describe.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_EPS = 1e-12


def row_log_loss(y_true: pd.Series, proba: pd.Series) -> pd.Series:
    """Log loss of each single prediction (large = confident and wrong).

    ``proba`` is P(up). Probabilities are clipped away from 0 and 1 so a
    fully confident mistake gives a big but finite number.
    """
    p = proba.clip(_EPS, 1 - _EPS)
    loss: pd.Series = -(y_true * np.log(p) + (1 - y_true) * np.log(1-p))  
    return loss


def build_error_table(
    proba: pd.Series,
    y_true: pd.Series,
    next_ret: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """One row per predicted day, with everything needed for error analysis.

    Parameters
    ----------
    proba : pd.Series
        P(up) for each day, from an out-of-sample model.
    y_true : pd.Series
        Actual 0/1 direction of the next day, same index as ``proba``.
    next_ret : pd.Series
        Actual next-day return as a fraction (0.01 = +1%), same index.
    threshold : float
        ``pred`` is 1 when ``proba >= threshold``.

    Returns
    -------
    pd.DataFrame
        Columns: proba, y_true, next_ret, pred, correct, confidence
        (distance of proba from 0.5), loss (per-row log loss).
    """
    if not (proba.index.equals(y_true.index) and proba.index.equals(next_ret.index)):
        raise ValueError("proba, y_true and next_ret must share the same index")

    table = pd.DataFrame(
        {
            "proba": proba,
            "y_true": y_true.astype(int),
            "next_ret": next_ret,
        }
    )
    table["pred"] = (table["proba"] >= threshold).astype(int)    
    table["correct"] = table["pred"] == table["y_true"]          
    table["confidence"] = (table["proba"] - 0.5).abs()
    table["loss"] = row_log_loss(table["y_true"], table["proba"])
    return table


def worst_predictions(table: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` predictions with the highest log loss, worst first."""
    return table.nlargest(n, "loss")                            


def group_metrics(table: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Prediction quality and strategy outcome for each group of days.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``build_error_table``.
    by : pd.Series
        Group label per day (e.g. a regime), same index as ``table``. Days
        with a missing label are left out.

    Returns
    -------
    pd.DataFrame
        One row per group:
        n, up_rate (share of up days), accuracy, majority_acc (accuracy of
        always guessing the more common direction), edge (accuracy minus
        majority_acc), acc_ci95 (rough +/- half-width of a 95% interval for
        accuracy), auc (NaN if the group has one class only), signal_rate
        (share of days predicted up), strat_sum and bench_sum (sum of simple
        next-day returns when following the signal / always holding; gross of
        costs and not compounded).
    """
    if not by.index.equals(table.index):
        raise ValueError("by must have the same index as table")

    rows: dict[object, dict[str, float]] = {}
    for name, group in table.groupby(by, observed=True):
        n = len(group)
        up_rate = float(group["y_true"].mean())
        accuracy = float(group["correct"].mean())
        majority = max(up_rate, 1-up_rate)                             
        one_class = group["y_true"].nunique() < 2                 
        auc = (
            float("nan")
            if one_class
            else float(roc_auc_score(group["y_true"], group["proba"]))
        )
        rows[name] = {
            "n": float(n),
            "up_rate": up_rate,
            "accuracy": accuracy,
            "majority_acc": majority,
            "edge": accuracy - majority,
            "acc_ci95": 1.96 * float(np.sqrt(accuracy * (1 - accuracy) / n)),
            "auc": auc,
            "signal_rate": float(group["pred"].mean()),
            "strat_sum": float((group["pred"] * group["next_ret"]).sum()),
            "bench_sum": float(group["next_ret"].sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def trend_regime(close: pd.Series, window: int = 200) -> pd.Series:
    """Label each day 'above SMA<window>' or 'below SMA<window>'.

    Uses only prices up to that day. The first ``window - 1`` days have no
    moving average yet and get NaN.
    """
    sma = close.rolling(window).mean()
    labels = np.where(close > sma, f"above SMA{window}", f"below SMA{window}")
    regime: pd.Series = pd.Series(labels, index=close.index).where(sma.notna())
    return regime


def volatility_regime(returns: pd.Series, window: int = 20) -> pd.Series:
    """Label each day 'low vol', 'mid vol' or 'high vol' (terciles).

    Volatility is the rolling standard deviation of the last ``window``
    returns. The tercile cut-offs are computed over the whole series passed
    in, so pass only the period you are analysing. The first ``window``
    days get NaN.
    """
    vol = returns.rolling(window).std()
    regime: pd.Series = pd.qcut(vol, 3, labels=["low vol", "mid vol", "high vol"])
    return regime