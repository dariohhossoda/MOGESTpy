import math


class zero_dimensional:
    """
    Classe de Qualidade da Água em Reservatórios - Modelo 0D
    Zero-Dimensional
    """
    def __init__(self, volume, Qin, Qout, Cin, As, k, v, timestep):
        """
        TODO: Documentar
        """
        self.Input = zero_dimensional.Input(volume,
                                           Qin,
                                           Qout,
                                           Cin,
                                           As)
        self.SimCount = len(self.Input.Volume)
        self.Output = zero_dimensional.Output(self.SimCount)
        self.Params = zero_dimensional.Params(k, v)
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
        _in = self.Input
        _out = self.Output
        _param = self.Params
        dt = self.Timestep

        _out.ConcentrationOut[0] = 0

        for i in range(1, self.SimCount):
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


class buildup_washoff:
    def __init__(self, use_name, Bmax, Nb, Kb,
                 threshold_flow, Nw, Kw, BuMethod, WoMethod,
                 timestep_h, initial_buildup, area, area_fraction,
                 aggregate = False):
        """_summary_

        Args:
            use_name (_type_): LandUse
            Bmax (_type_): Maximum Buildup, normalized (kg/km²)
            Nb (_type_): Buildup Exponent  (-). Nb <= 1
            Kb (_type_): Buildup rate constant.
            - Pow: kg/km²*d^-Nb;
            - Exp: d^-1;
            - Sat: d
        """
        self.use_name = use_name
        self.Bmax = Bmax
        self.Nb = Nb
        self.Kb = Kb
        self.Kw = Kw
        self.Nw = Nw
        self.timestep_h = timestep_h
        self.timestep_d = self.timestep_d / 24
        self.Washoff = []
        self.BuildUp = []
        self.EffectiveWashoff = []
        self.Aggregate = aggregate

    # region tempo (time) das equações de BuildUp
    def TimeFromBuildUpPow(self, buildup):
        try:
            time = (buildup) ** (1 / self.Nb)
        except:
            time = 0

        return time

    def TimeFromBuildUpExp(self, buildup):
        try:
            time = -math.log(1 - buildup / self.Bmax) / self.Kb
        except:
            time = 0

        return time

    def TimeFromBuildUpSat(self, buildup):
        try:
            time = buildup * self.Kb / (self.Bmax - buildup)
        except:
            time = 0

        return time
    # endregion

    # region BuildUp
    def BuildUpPow(self, time):
        return math.min(self.Bmax, self.Kb * (time ** self.Nb))

    def BuildUpExp(self, time):
        return self.Bmax * (1 - math.e ** (self.Kb * time))

    def BuildUpSat(self, time):
        return self.Bmax * time / (self.Kb + time)
    # endregion

    # region Washoff
    def WashoffExp(self, surface_flow, buildup_mass):
        return self.Kw * (surface_flow ** self.Nw) * buildup_mass

    def WashoffRating(self, surface_flow, watershed_area):
        return self.Kw * (1000 * surface_flow * watershed_area) ** self.Nw

    def WashoffEMC(self, surface_flow, watershed_area):
        return self.Kw * surface_flow * watershed_area
    # endregion

    def Process(simulation):
        """
        simulation -> BuildUpWashoff
        """
        raise NotImplementedError('Ainda não implementado!')
