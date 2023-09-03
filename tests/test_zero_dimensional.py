from mogestpy.quality import zero_d

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
    
    reactor = zero_d.ZeroDimensional(V, Qin, Qout, Cin, As, k, v, dt)
    reactor.RunModel()
    
    assert reactor.Output.ConcentrationOut[-1] == 23.999999566320213
