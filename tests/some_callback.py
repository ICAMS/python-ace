def test_external_callback(basis_config, current_fit_iteration, current_fit_cycle, current_ladder_step):
    print("Inside external callback")
    with open("external_callback.dat","a") as f:
        print((current_fit_iteration, current_fit_cycle, current_ladder_step), file=f)
