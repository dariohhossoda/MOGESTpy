from mogestpy.quantity.Hydrological import SMAP2


def test_smap2_init():
    """
    Tests initialization of SMAP2 class
    """
    smap = SMAP2.Smap()
    assert smap.i == 0


def test_smap2_discharge_calc():
    """
    Tests Discharge Calculation of SMAP2 class
    """
    smap = SMAP2.Smap()
    assert abs(smap.discharge_calc(
        0.0, 0.991415730337079, 17800) - 204.25) < 1e-6


def test_smap2_transfer_functions():
    """
    Tests Transfer Functions of SMAP2 class
    """
    smap = SMAP2.Smap()
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
    smap = SMAP2.Smap()
    smap.Str = 165.060822610565
    smap.Ad = 17800.0
    smap.Crec = 20.0
    smap.Tuin = 0.8
    smap.Ebin = 204.25
    smap.Capc = 40.0
    smap.kkt = 120.0
    smap.k2t = 7.97306669390268
    smap.Ai = 4

    prec = 21.4017248169365
    etp = 2.4
    Rsolo = 80.9265336890574
    Es_ref = 2.982390340677
    Er_ref = 2.982390340677
    Tu = 0.490283111456380000
    Es_ = smap.Es_calc(prec, smap.Ai, smap.Str, Rsolo)

    assert abs(Es_ - Es_ref) < 1e-6
    assert abs(smap.Er_calc(prec, etp, Es_, Tu) - Er_ref) < 1e-6


def test_smap2_reservoir_init():
    """
    Tests Rsolo_calc method of SMAP2 class
    """
    smap = SMAP2.Smap()
    assert abs(smap.Rsolo0(0.8, 165.060822610565) - 132.048658) < 1e-6
    assert smap.RSup0() == 0
    assert abs(smap.RSub0(204.25, 120, 17800) - 172.133452) < 1e-6


def test_smap2_run_step():
    """
    Tests RunStep method of SMAP2 class
    """
    smap = SMAP2.Smap()
    smap.Str = 165.060822610565
    smap.Ad = 17800.0
    smap.Crec = 20.0
    smap.Tuin = 0.8
    smap.Ebin = 204.25
    smap.Capc = 40.0
    smap.kkt = 120.0
    smap.k2t = 7.97306669390268
    smap.Ai = 4.0

    assert abs(smap.RunStep(0, 2.4) - 204.250000) < 1e-6


def test_smap2_run_step_multiple():
    """
    Tests RunStep method of SMAP2 class
    """
    smap = SMAP2.Smap()
    smap.Str = 165
    smap.Ad = 17800
    smap.Crec = 20
    smap.Tuin = 0.8
    smap.Ebin = 204
    smap.Capc = 40
    smap.kkt = 120
    smap.k2t = 7.97

    assert round(smap.RunStep(0, 2.4), 6) != 0
    # assert smap.RunStep(0, 2.4) - 216 < 1e-6
    # assert smap.RunStep(0, 2.4) - 224 < 1e-6
    # assert smap.RunStep(0, 2.4) - 224 < 1e-6
