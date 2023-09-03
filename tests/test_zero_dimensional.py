from mogestpy.quality import zero_dimensional


def test_zero_dimensional_1():
    vector_size = 10
    Qin = [50 for _ in range(vector_size)]
    Qout = [50 for _ in range(vector_size)]
    Cin = [60 for _ in range(vector_size)]
    V = [1000 for _ in range(vector_size)]
    As = [50 for _ in range(vector_size)]
    v = [.5 for _ in range(vector_size)]
    k = [.05 for _ in range(vector_size)]

    dt = 50
    
    reactor = zero_dimensional(V, Qin, Qout, Cin, As, k, v, dt)
    reactor.RunModel()
    
    assert reactor.Output.ConcentrationOut == [0,
                                               20.689655172413794,
                                               23.543400713436384,
                                               23.937020788060188,
                                               23.99131321214623,
                                               23.998801822364996,
                                               23.99983473411931,
                                               23.99997720470611,
                                               23.999996855821532,
                                               23.999999566320213]
    