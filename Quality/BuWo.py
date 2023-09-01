import math
import numpy as np

class BuildUpWashoff:
    def __init__(self, landuse_name, Bmax, Nb, Kb,
                 threshold_flow, Nw, Kw, BuMethod, WoMethod,
                 timestep_h, initial_buildup, area, area_fraction,
                 surface_flow, aggregate = False):

        self.LanduseName = landuse_name
        
        self.Bmax = Bmax
        self.Nb = Nb
        self.Kb = Kb
        self.Kw = Kw
        self.Nw = Nw
        
        self.timestep_h = timestep_h
        self.timestep_d = self.timestep_h / 24
        
        self.Washoff = []
        self.BuildUp = []
        self.EffectiveWashoff = []
        
        self.Aggregate = aggregate
        self.InitialBuildUp = initial_buildup
        
        self.Area = area
        self.AreaFraction = area_fraction
        
        self.SurfaceFlow = surface_flow
        self.ThresholdFlow = threshold_flow
        
        self.BuMethod = BuMethod
        self.WoMethod = WoMethod

    # region Time from BuildUp Equations
    def TimeFromBuildUpPow(self, buildup):
        try:
            time = (buildup / self.Kb) ** (1 / self.Nb)
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
    # endregion Time from BuildUp Equations

    # region BuildUp Equations
    def BuildUpPow(self, time):
        return math.min(self.Bmax, self.Kb * (time ** self.Nb))

    def BuildUpExp(self, time):
        return self.Bmax * (1 - math.e ** (self.Kb * time))

    def BuildUpSat(self, time):
        return self.Bmax * time / (self.Kb + time)
    # endregion BuildUp Equations

    # region Washoff Equations
    def WashoffExp(self, surface_flow, buildup_mass):
        return self.Kw * (surface_flow ** self.Nw) * buildup_mass

    def WashoffRating(self, surface_flow, watershed_area):
        return self.Kw * (1000 * surface_flow * watershed_area) ** self.Nw

    def WashoffEMC(self, surface_flow, watershed_area):
        return self.Kw * surface_flow * watershed_area
    # endregion Washoff Equations

    def Process(self):
        """
        Roda o modelo de acúmulo e lavagem (BuildUp e Washoff).
        """
        buildup = self.InitialBuildUp / (self.Area * self.AreaFraction)
        
        time_from_buildup = {1 : self.TimeFromBuildUpPow(buildup),
                             2 : self.TimeFromBuildUpPow(buildup),
                             3 : self.TimeFromBuildUpPow(buildup)}
        
        bu_time = time_from_buildup.get(self.BuMethod)
        
        for i in range(len(self.Washoff)):
            if self.SurfaceFlow[i] < self.ThresholdFlow:
                bu_time += self.timestep_d
                self.Washoff[i] = 0

                buildup_curve = {1 : self.BuildUpPow(bu_time),
                                 2 : self.BuildUpExp(bu_time),
                                 3 : self.BuildUpSat(bu_time)}

                buildup_specific = buildup_curve.get(self.BuMethod)

                buildup_mass = buildup_specific * self.Area * self.AreaFraction
            else:
                washoff_curve = {1 : self.WashoffExp(self.SurfaceFlow, buildup_mass),
                                 2 : self.WashoffRating(self.SurfaceFlow, buildup_mass),
                                 3 : self.WashoffEMC(self.SurfaceFlow, buildup_mass)}

                washoff_rate = washoff_curve.get(self.WoMethod)

                buildup_specific = 0
                self.Washoff[i] = washoff_rate * self.timestep_d
                self.EffectiveWashoff[i] = math.min(self.Washoff[i], buildup_mass)
                buildup_mass -= self.EffectiveWashoff[i]

                bu_time = time_from_buildup.get(self.BuMethod)

            self.BuildUp[i] = buildup_mass
