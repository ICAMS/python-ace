import numpy as np
import pandas as pd
from pyace import *
from pyace.linearacefit import LinearACEFit, LinearACEDataset

df_name = 'tests/representative_df.pckl.gzip'
df2_name = "tests/df_AlNi(murn).pckl.gzip"
# Create empty bbasis configuration
bconf = create_multispecies_basis_config(potential_config={
    "deltaSplineBins": 0.001,
    "elements": ['Al'],

    "embeddings": {
        "ALL": {
            "npot": 'FinnisSinclairShiftedScaled',
            "fs_parameters": [1, 1],
            "ndensity": 1,
        },
    },

    "bonds": {
        "ALL": {
            "radbase": "SBessel",
            "radparameters": [5.25],
            "rcut": 6,
            "dcut": 0.01,
        }
    },

    "functions": {
        # "number_of_functions_per_element": 1000,
        "ALL": {
            "nradmax_by_orders": [6, 5, 4, 3, 2],
            "lmax_by_orders": [0, 4, 3, 2, 1]}
    }
}
)

bconf2 = create_multispecies_basis_config(potential_config={
    "deltaSplineBins": 0.001,
    "elements": ['Al', 'Ni'],

    "embeddings": {
        "ALL": {
            "npot": 'FinnisSinclairShiftedScaled',
            "fs_parameters": [1, 1],
            "ndensity": 1,
        },
    },

    "bonds": {
        "ALL": {
            "radbase": "SBessel",
            "radparameters": [5.25],
            "rcut": 6,
            "dcut": 0.01,
        }
    },

    "functions": {
        # "number_of_functions_per_element": 1000,
        "ALL": {
            "nradmax_by_orders": [6, 2, 1],
            "lmax_by_orders": [0, 2, 1]}
    }
}
)


def test_LinearACEDataset_singlespecie():
    df = pd.read_pickle(df_name, compression='gzip')
    print("df.shape=", df.shape)
    assert len(df) == 6
    linear_ds = LinearACEDataset(bconf, df)
    dm = linear_ds.get_design_matrix(max_workers=2, verbose=True)
    print("dm.shape=", dm.shape)
    assert dm.shape == (462, 715)


def test_LinearACEDataset_multispecie():
    print("bconf2.total_number_of_functions=", bconf2.total_number_of_functions)
    assert bconf2.total_number_of_functions == 104
    df = pd.read_pickle(df2_name, compression='gzip')
    # df = df.sample(n=10, random_state=42)
    print("df.shape=", df.shape)
    assert len(df) == 174
    linear_ds = LinearACEDataset(bconf2, df)
    dm = linear_ds.get_design_matrix(max_workers=2, verbose=True)
    print("dm.shape=", dm.shape)
    assert dm.shape == (1521, 104)


def test_LinearACEFit():
    print("bconf2.total_number_of_functions=", bconf2.total_number_of_functions)
    assert bconf2.total_number_of_functions == 104
    df = pd.read_pickle(df2_name, compression='gzip')
    print("df.shape=", df.shape)
    assert len(df) == 174
    ds = LinearACEDataset(bconf2, df)

    linear_fit = LinearACEFit(train_dataset=ds)
    linear_fit.fit()
    errors = linear_fit.compute_errors(ds)
    print("errors=", errors)
    ref_errors = {'epa_mae': 0.003831128214766918, 'epa_rmse': 0.005182476661323097,
                  'f_comp_mae': 0.0012976990221506717, 'f_comp_rmse': 0.0035144969263983432}
    for k, v in ref_errors.items():
        assert np.isclose(errors[k], v)

    # get basis
    basis = linear_fit.get_bbasis()
    calc = PyACECalculator(basis)
    e_pred, f_pred = linear_fit.predict(ds, reshape_forces=True)
    # take first  ase_atoms
    at = df.iloc[0]["ase_atoms"].copy()
    at.set_calculator(calc)
    at.get_potential_energy()
    assert np.allclose(at.get_potential_energy() / len(at), e_pred[0])
    assert np.allclose(at.get_forces(), f_pred[:len(at)])
