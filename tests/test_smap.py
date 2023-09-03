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

    size = 10

    P = [10 for _ in range(size)]
    ET = [2 for _ in range(size)]

    bacia = SMAP.SMAP.Basin(Str, AD, Crec, TUin, EBin, Capc, kkt, k2t, Ai)
    ponto = SMAP.SMAP.Point(P, ET)
    modelo = SMAP.SMAP(ponto, bacia)
    modelo.RunModel()

    assert round(modelo.Q[-1], 3) == 92.502