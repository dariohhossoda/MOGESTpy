class ZeroDimensional:
    """
    Classe de Qualidade da Água em Reservatórios - Modelo 0D
    Zero-Dimensional
    """
    def __init__(self, volume, Qin, Qout, Cin, As, k, v, timestep):
        """
        TODO: Documentar
        """
        print("init")
        self.Input = ZeroDimensional.Input(volume,
                                           Qin,
                                           Qout,
                                           Cin,
                                           As)
        self.SimCount = len(self.Input.Volume)
        self.Output = ZeroDimensional.Output(self.SimCount)
        self.Params = ZeroDimensional.Params(k, v)
        self.Timestep = timestep
    class Output:
        """
        Saída do modelo 0D

        Cout - Concentração de saída após o término da simulação
        """
        def __init__(self, size):
            self.ConcentrationOut = [0 for _ in range(size)]

    class Input:
        """
        Inputs do modelo 0D
        ---------------
        Volume (V) - list

        Vazão de montante (Qin) - list

        Vazão de jusante (Qout) - list

        Concentração de montante (Cin) - list

        Carga de montante (Cin) - list

        Área de contato (As) - list
        """
        def __init__(self,
                     volume,
                     inflow,
                     outflow,
                     concentration_in,
                     contact_area):

            self.Volume = volume
            self.Inflow = inflow
            self.Outflow = outflow
            self.ConcentrationIn = concentration_in
            self.ContactArea = contact_area

    class Params:
        """
        Parâmetros do modelo 0D

        Coecifiente de Reação - k
        Velocidade de Assentamento - v
        """
        def __init__(self, k, v):
            self.ReactionCoefficient = k
            self.SettlingVelocity = v

    def RunModel(self):
        print("1")
        _in = self.Input
        _out = self.Output
        _param = self.Params
        dt = self.Timestep

        _out.ConcentrationOut[0] = 0

        for i in range(1, self.SimCount):
            print(i)
            _out.ConcentrationOut[i] = (_out.ConcentrationOut[i - 1]
                                        + dt / _in.Volume[i]
                                        * _in.Inflow[i]
                                        * _in.ConcentrationIn[i]
                                        ) / (1 + dt / _in.Volume[i]
                                             * (_in.Outflow[i]
                                                + _param.ReactionCoefficient[i]
                                                * _in.Volume[i]
                                                + _param.SettlingVelocity[i]
                                                * _in.ContactArea[i]
                                                + (_in.Volume[i]
                                                   - _in.Volume[i - 1]) / dt))
            print(_out.ConcentrationOut[-1])


def test_zero_dimensional_1():
    print("1")
    vector_size = 10
    Qin = [50 for _ in range(vector_size)]
    Qout = [50 for _ in range(vector_size)]
    Cin = [60 for _ in range(vector_size)]
    V = [1000 for _ in range(vector_size)]
    As = [50 for _ in range(vector_size)]
    v = [.5 for _ in range(vector_size)]
    k = [.05 for _ in range(vector_size)]

    dt = 50
    
    reactor = ZeroDimensional(V, Qin, Qout, Cin, As, k, v, dt)
    print("ok")
    reactor.RunModel()
    print(reactor.Output.ConcentrationOut[-1])
    ''
    assert round(reactor.Output.ConcentrationOut[-1], 3) == 25.0
    print("assert ok")
test_zero_dimensional_1()