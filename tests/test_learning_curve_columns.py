"""The learning curve must find the TFT's train loss under Lightning's naming."""

import pandas as pd

from scripts.plot_learning_curve import _epoch_curve


def _metrics(**columns):
    return pd.DataFrame({"epoch": [0, 0, 1, 1], **columns})


def test_the_tft_epoch_level_train_loss_is_found_under_the_plain_name():
    df = _metrics(
        train_loss_step=[1.2, None, 0.7, None],
        train_loss_epoch=[None, 1.0, None, 0.5],
        val_loss=[None, 0.9, None, 0.4],
    )

    assert _epoch_curve(df, "train_loss").tolist() == [1.0, 0.5]


def test_the_lstm_plain_column_is_preferred_when_present():
    df = _metrics(train_loss=[None, 2.0, None, 1.0], train_loss_epoch=[None, 9.0, None, 9.0])

    assert _epoch_curve(df, "train_loss").tolist() == [2.0, 1.0]


def test_a_metric_that_was_never_logged_is_none():
    assert _epoch_curve(_metrics(val_loss=[None, 0.9, None, 0.4]), "train_loss") is None
