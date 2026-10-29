import numpy as np
import pandas as pd
import pytest

from pyace.metrics_aggregator import FitMetrics


@pytest.mark.parametrize("shape", [(4,), (4, 1)])
def test_regularization_contributions_are_scalars(shape):
    metrics = FitMetrics(
        w_e=np.ones((1, 1)), w_f=np.ones((1, 1)),
        e_scale=0.5, f_scale=0.5, ncoefs=2,
        regs=np.array([2.0, 3.0, 4.0, 5.0]).reshape(shape),
    )
    metrics.loss = 0.0
    metrics.eval_time = 0.0
    metrics.reg_weights = np.array([0.1, 0.2, 0.3, 0.4]).reshape(shape)
    metrics.compute_metrics(
        de=np.zeros((1, 1)), de_pa=np.zeros((1, 1)),
        df=np.zeros((1, 3)), nat=np.ones((1, 1), dtype=int),
        dataframe=pd.DataFrame({"energy_corrected": [0.0]}),
    )

    result = metrics.to_FitMetricsDict()

    assert result["l1_reg_contrib"] == pytest.approx(0.2)
    assert result["l2_reg_contrib"] == pytest.approx(0.6)
    assert result["extra_regularization_contrib"] == pytest.approx([1.2, 2.0])
    assert type(result["l1_reg_contrib"]) is float
    assert type(result["l2_reg_contrib"]) is float
    assert all(type(value) is float for value in result["extra_regularization_contrib"])
