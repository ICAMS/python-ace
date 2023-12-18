import pytest

from pyace.basis import BBasisFunctionSpecification, ACEBBasisFunction


def test_constructor_from_bBasisFunctionSpecification():
    spec = BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[1, 1], coeffs=[1.0])
    spec2 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al", "Al"], ns=[2, 3, 3, 3],
                                        ls=[1, 1, 2, 2], LS=[2, 2], coeffs=[1.0, 2])
    func = ACEBBasisFunction(spec, is_half_basis=False, compress=False)
    func2 = ACEBBasisFunction(spec2, compress=False)

    func.print()
    func2.print()
    assert func.num_ms_combs == 3
    assert func.rank == 2
    assert func.rankL == 0
    assert func.is_proxy == False
    assert func.is_half_ms_basis == False
    assert func.ns == [1, 1]
    assert func.ls == [1, 1]
    assert func.coeffs == [1.]

    assert func2.num_ms_combs == 19
    assert func2.rank == 4
    assert func2.rankL == 2
    assert func2.is_proxy == False
    assert func2.is_half_ms_basis == True
    assert func2.LS == [2, 2]


def test_constructor_from_bBasisFunctionSpecification_compress():
    spec = BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[1, 1], coeffs=[1.0])
    spec2 = BBasisFunctionSpecification(["Al", "Al", "Al", "Al", "Al"], ns=[2, 3, 3, 3],
                                        ls=[1, 1, 2, 2], LS=[2, 2], coeffs=[1.0, 2])
    func = ACEBBasisFunction(spec, False, compress=True)
    func2 = ACEBBasisFunction(spec2, compress=True)

    func.print()
    func2.print()
    assert func.num_ms_combs == 2
    assert func.rank == 2
    assert func.rankL == 0
    assert func.is_proxy == False
    assert func.is_half_ms_basis == False
    assert func.ns == [1, 1]
    assert func.ls == [1, 1]
    assert func.coeffs == [1.]

    assert func2.num_ms_combs == 12
    assert func2.rank == 4
    assert func2.rankL == 2
    assert func2.is_proxy == False
    assert func2.is_half_ms_basis == True
    assert func2.LS == [2, 2]


def test_constructor_bBasisFunctionSpecification_from_ACEBBasisFunction():
    spec = BBasisFunctionSpecification(["Al", "Al", "Al"], ns=[1, 1], ls=[1, 1], coeffs=[1.0])
    func = ACEBBasisFunction(spec)
    new_spec = BBasisFunctionSpecification(["Al"], func)
    assert new_spec.ns == [1, 1]
    assert new_spec.ls == [1, 1]
    assert new_spec.coeffs == [1.0]
    assert new_spec.elements == ["Al", "Al", "Al"]


def test_sort_order():
    # {type: Al Al Al Al, nr: [1, 1, 1], nl: [2, 1, 1], lint: [1]
    spec = BBasisFunctionSpecification(["Al", "Al", "Al", "Al"], ns=[1, 1, 1], ls=[2, 1, 1], LS=[1], coeffs=[1.0])
    func = ACEBBasisFunction(spec)
    print("func.sort_order=", func.sort_order)
    assert func.sort_order == [1, 2, 0]


def test_constructor_from_bBasisFunctionSpecification_multispecies_exception():
    spec = BBasisFunctionSpecification(["Al", "Al", "Ni"], ns=[1, 1], ls=[1, 1], coeffs=[1.0])
    with pytest.raises(ValueError):
        func = ACEBBasisFunction(spec)


def test_constructor_from_bBasisFunctionSpecification_multispecies():
    spec = BBasisFunctionSpecification(["Ni", "Al", "Ni"], ns=[1, 1], ls=[1, 1], coeffs=[1.0])
    elements_mapping = {"Al": 0, "Ni": 1}
    func = ACEBBasisFunction(spec, elements_mapping)
    func.print()
    assert func.num_ms_combs == 2
    assert func.rank == 2
    assert func.rankL == 0
    assert func.is_proxy == False
    assert func.is_half_ms_basis == True
    assert func.mu0 == 1
    assert func.mus == [0, 1]
    assert func.ns == [1, 1]
    assert func.ls == [1, 1]
    assert func.coeffs == [1.]
