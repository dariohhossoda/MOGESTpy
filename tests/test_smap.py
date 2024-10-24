from mogestpy.quantity.Hydrological import SMAP


def test_smap_1():
    Str = 1000
    AD = 1000
    Crec = 20
    TUin = 100/100
    EBin = 0
    Capc = 30
    kkt = 30
    k2t = 1
    Ai = 2.5

    data_size = 10

    prec = [10 for _ in range(data_size)]
    etp = [2 for _ in range(data_size)]

    bacia = SMAP.SMAP.Basin(Str, AD, Crec, TUin, EBin, Capc, kkt, k2t, Ai)
    ponto = SMAP.SMAP.Point(prec, etp)
    modelo = SMAP.SMAP(ponto, bacia)
    modelo.RunModel()

    assert abs(modelo.Q[-1] - 0.6965439607456360) < 1e-6
