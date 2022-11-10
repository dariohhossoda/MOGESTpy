import math


class BuildUpWashoff:
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
