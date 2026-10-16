"""Walk-forward (rolling-window) out-of-sample predictions.

A single train/validation split gives one model and one test period. A
walk-forward run instead re-trains the model again and again as time moves
on, and every prediction comes from a model that only saw the past, which is
how a model would actually be used in practice.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

logger = logging.getLogger(__name__)


def walk_forward_proba(
    data: pd.DataFrame,
    model: BaseEstimator,
    feature_cols: list[str],
    target_col: str = "direction",
    train_size: int = 250,
    step: int = 20,
    gap: int = 1,
    rolling: bool = True,
) -> pd.Series:
    """Predict P(up) for every row after the first training window.

    The data is walked through in blocks of ``step`` rows. For each block a
    fresh copy of ``model`` is fitted on rows that come strictly before the
    block, then predicts the block.

    Parameters
    ----------
    data : pd.DataFrame
        Features and target, sorted by date ascending, no NaN.
    model : BaseEstimator
        Unfitted scikit-learn classifier (or Pipeline) with ``predict_proba``.
        It is cloned for every block and never modified.
    feature_cols : list[str]
        Columns used as inputs. The target must not be among them.
    target_col : str
        Column with the 0/1 label.
    train_size : int
        Rows in the training window. With ``rolling=True`` the window always
        has exactly this many rows; with ``rolling=False`` it is only the
        size of the *first* window and the window then keeps growing.
    step : int
        Rows predicted per refit (about 20 is one trading month).
    gap : int
        Rows skipped between the end of the training window and the block.
        Use at least the label horizon so no training label depends on
        prices at or after the block start.
    rolling : bool
        True for a fixed-length sliding window, False for an expanding one.

    Returns
    -------
    pd.Series
        Probabilities named ``proba``, indexed by the predicted dates. The
        first ``train_size + gap`` rows of ``data`` get no prediction.
    """
    if step < 1:
        raise ValueError("step must be at least 1")
    if gap < 0:
        raise ValueError("gap must not be negative")
    if train_size < 2:
        raise ValueError("train_size must be at least 2")
    if target_col in feature_cols:
        raise ValueError("target_col must not be one of feature_cols")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data must be sorted by date ascending")

    first_block = train_size + gap
    if first_block >= len(data):
        raise ValueError(
            f"need more than train_size + gap = {first_block} rows, got {len(data)}"
        )

    pieces: list[pd.Series] = []
    for block_start in range(first_block, len(data), step):          
        block_end = min(block_start + step, len(data))

        train_end = block_start - gap                               
        train_begin = train_end - train_size if rolling else 0    

        train = data.iloc[train_begin:train_end]
        block = data.iloc[block_start:block_end]                        
        y_train = train[target_col].astype(int)

        if y_train.nunique() < 2:
            # One-class window: the classifier cannot be fitted, fall back to
            # predicting that class with certainty.
            logger.warning("Training window ending %s has one class", train.index[-1])
            proba = np.full(len(block), float(y_train.iloc[0]))
        else:
            fitted = clone(model).fit(train[feature_cols], y_train)  
            proba = fitted.predict_proba(block[feature_cols])[:, 1] 

        pieces.append(pd.Series(proba, index=block.index))

    result: pd.Series = pd.concat(pieces).rename("proba")
    return result