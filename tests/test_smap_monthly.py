from mogestpy.quantity.hydrological.SMAPm import SmapM


def test_smapm_init():
    """
    Tests initialization of SmapM class
    """
    smap = SmapM()
    assert smap.i == 0


def test_smapm_discharge_calc():
    """
    Tests discharge calculation of SmapM class
    """
    smap = SmapM()
    assert abs(smap.Q_calc(0.0, 0.991415730337079, 17800) - 6.709961977186313) < 1e-6


def test_smapm_transfer_functions():
    """
    Tests transfer functions of SmapM class
    """
    smap = SmapM()

    assert smap.Es_calc(0, 0.8, 2.5) == 0


def test_smapm_transfer_functions2():
    """
    Tests transfer functions of SmapM class
    """
    smap = SmapM(
        Str=165.060822610565,
        Ad=17800.0,
        Crec=0.2,
        Tuin=0.8,
        Ebin=204.25,
        Pes=4.0,
        kkt=1.5,
    )

    prec = 21.0
    etp = 2.4

    Rsolo = 80.9265337
    Tu = 0.4902831

    Es_ref = 1.2134083758267047
    Er_ref = 1.17667944

    Es_calc = smap.Es_calc(prec, Tu, smap.Pes)
    Er_calc = smap.Er_calc(Tu, etp)

    Rec_calc = smap.Rec_calc(smap.Crec, Tu, Rsolo)
    Rec_ref = 0.9352088935066863

    assert abs(Es_calc - Es_ref) < 1e-6, f"Es: {Es_calc} | {Es_ref}"
    assert abs(Er_calc - Er_ref) < 1e-6, f"Er: {Er_calc} | {Er_ref}"
    assert abs(Rec_calc - Rec_ref) < 1e-6, f"Rec: {Rec_calc} | {Rec_ref}"


def test_smapm_reservoir_init():
    """
    Tests reservoir initialization methods of SmapM class
    """
    smap = SmapM()
    assert abs(smap.Rsolo0_calc(0.8, 165.060822610565) - 132.04865808845202) < 1e-6
    assert abs(smap.Rsub0_calc(204.25, 1.5, 17800) - 81.55484284931562) < 1e-6


def test_smapm_run_step():
    """
    Tests run_step method of SmapM class
    """
    smap = SmapM(
        Str=165.060822610565,
        Ad=17800.0,
        Crec=0.2,
        Tuin=0.8,
        Ebin=204.25,
        Pes=4.0,
        kkt=1.5,
    )

    smap.Tu = smap.Tu_calc(smap.Rsolo, smap.Str)
    assert abs(smap.Tu - 0.8) < 1e-6, f"Tu: {smap.Tu}"
    assert abs(smap.run_step(0, 2.4) - 204.250000) < 1e-6


def test_smapm_run_step_multiple():
    """
    Tests run_step method of SmapM class for multiple timesteps
    """
    smap = SmapM(
        Str=165.060822610565,
        Ad=17800.0,
        Crec=0.2,
        Tuin=0.8,
        Ebin=204.25,
        Pes=4.0,
        kkt=1.5,
    )

    step_1 = smap.run_step(0, 2.4)
    step_2 = smap.run_step(0, 2.4)
    step_3 = smap.run_step(0, 2.4)

    assert abs(step_1 - 204.25) < 1e-6, f"Step 1: {step_1}"
    assert abs(step_2 - 155.76113647133056) < 1e-6, f"Step 2: {step_2}"
    assert abs(step_3 - 114.43783605128543) < 1e-6, f"Step 3: {step_3}"
