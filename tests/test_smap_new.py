from mogestpy.quantity.Hydrological.SMAP2 import Smap


def test_smap2_init():
    """
    Tests initialization of SMAP2 class
    """
    smap = Smap()
    assert smap.i == 0


def test_smap2_discharge_calc():
    """
    Tests Discharge Calculation of SMAP2 class
    """
    smap = Smap()
    assert abs(smap.discharge_calc(
        0.0, 0.991415730337079, 17800) - 204.25) < 1e-6


def test_smap2_transfer_functions():
    """
    Tests Transfer Functions of SMAP2 class
    """
    smap = Smap()
    smap.Str = 165.060822610565
    smap.Ad = 17800.0
    smap.Crec = 20.0
    smap.Tuin = 0.8
    smap.Ebin = 204.25
    smap.Capc = 40.0
    smap.kkt = 120.0
    smap.k2t = 7.97306669390268
    smap.Ai = 2.5

    assert smap.Es_calc(0, smap.Ai, smap.Str, 132.048658088452) == 0


def test_smap2_transfer_functions2():
    """
    Tests Transfer Functions of SMAP2 class
    """
    smap = Smap(
        Str=165.060822610565,
        Ad=17800.0,
        Crec=20.0,
        Tuin=0.8,
        Ebin=204.25,
        Capc=40.0,
        kkt=120.0,
        k2t=7.97306669390268,
        Ai=4.0
    )

    prec = 21.0
    etp = 2.4

    Rsolo = 80.9265337
    Tu = 0.4902831

    Es_ref = 2.8575867
    Er_ref = 2.4

    Es_calc = smap.Es_calc(prec, smap.Ai, smap.Str, Rsolo)
    Er_calc = smap.Er_calc(prec, etp, Es_calc, Tu)

    Rec_calc = smap.Rec_calc(smap.Crec, Tu, Rsolo, smap.Capc, smap.Str)
    Rec_ref = 1.4612599

    assert abs(Es_calc - Es_ref) < 1e-6, f'Es: {Es_calc} | {Es_ref}'
    assert abs(Er_calc - Er_ref) < 1e-6, f'Er: {Er_calc} | {Er_ref}'
    assert abs(Rec_calc - Rec_ref) < 1e-6, f'Rec: {Rec_calc} | {Rec_ref}'


def test_smap2_reservoir_init():
    """
    Tests Rsolo_calc method of SMAP2 class
    """
    smap = Smap()
    assert abs(smap.Rsolo0(0.8, 165.060822610565) - 132.048658) < 1e-6
    assert smap.RSup0() == 0
    assert abs(smap.RSub0(204.25, 120, 17800) - 172.133452) < 1e-6


def test_smap2_run_step():
    """
    Tests RunStep method of SMAP2 class
    """
    smap = Smap(
        Str=165.060822610565,
        Ad=17800.0,
        Crec=20.0,
        Tuin=0.8,
        Ebin=204.25,
        Capc=40.0,
        kkt=120.0,
        k2t=7.97306669390268,
        Ai=4.0
    )

    smap.Tu = smap.Tu_calc(smap.Rsolo, smap.Str)
    assert abs(smap.Tu -
               0.8) < 1e-6, f"Tu: {smap.Tu}"
    assert abs(smap.RunStep(0, 2.4) - 204.250000) < 1e-6


def test_smap2_run_step_multiple():
    """
    Tests RunStep method of SMAP2 class
    """
    smap = Smap(
        Str=165,
        Ad=17800.0,
        Crec=20.0,
        Tuin=0.8,
        Ebin=204.25,
        Capc=40.0,
        kkt=120.0,
        k2t=8,
        Ai=4.0
    )

    step_1 = smap.RunStep(0, 2.4)
    step_2 = smap.RunStep(0, 2.4)
    step_3 = smap.RunStep(0, 2.4)

    assert abs(step_1 - 204.25) < 1e-6, f"Step 1: {step_1}"
    assert abs(step_2 - 215.6038845) < 1e-6, f"Step 2: {step_2}"
    assert abs(step_3 - 223.5623454) < 1e-6, f"Step 3: {step_3}"
