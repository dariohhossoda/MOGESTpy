from spotpy.objectivefunctions import kge, nashsutcliffe, rmse, pbias
from scipy.optimize import minimize, differential_evolution

class BuildUpWashoff:
    def __init__(self, landuse_name, Bmax, Nb, Kb,
                 threshold_flow, Nw, Kw, BuMethod, WoMethod,
                 timestep_h, initial_buildup, area, area_fraction,
                 surface_flow, aggregate=False):

        self.LanduseName = landuse_name

        self.Bmax = Bmax
        self.Nb = Nb
        self.Kb = Kb
        self.Kw = Kw
        self.Nw = Nw

        self.timestep_h = timestep_h
        self.timestep_d = self.timestep_h / 24

        self.Washoff = [0,0,0,0,0,0,0,0,0,0,0,0] #aqui o tamanho destes vetores deve ser alterado de modo a determinar o intervalo de tempo total simulado
        self.BuildUp = [0,0,0,0,0,0,0,0,0,0,0,0]
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
        return min(self.Bmax, self.Kb * (time ** self.Nb))

    def BuildUpExp(self, time):
        return self.Bmax * (1 - math.e ** (-self.Kb * time))

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

        time_from_buildup = {1: self.TimeFromBuildUpPow(buildup),
                             2: self.TimeFromBuildUpPow(buildup),
                             3: self.TimeFromBuildUpPow(buildup)}

        bu_time = time_from_buildup.get(self.BuMethod)

        for i in range(len(self.Washoff)):
            if self.SurfaceFlow[i] < self.ThresholdFlow:
                bu_time += self.timestep_d
                self.Washoff[i] = 0

                buildup_curve = {1: self.BuildUpPow(bu_time),
                                 2: self.BuildUpExp(bu_time),
                                 3: self.BuildUpSat(bu_time)}

                buildup_specific = buildup_curve.get(self.BuMethod)     #2

                buildup_mass = buildup_specific * self.Area * self.AreaFraction
            else:
                washoff_curve = {1: self.WashoffExp(self.SurfaceFlow, buildup_mass),
                                 2: self.WashoffRating(self.SurfaceFlow, buildup_mass),
                                 3: self.WashoffEMC(self.SurfaceFlow, buildup_mass)}

                washoff_rate = washoff_curve.get(self.WoMethod)

                buildup_specific = 0
                self.Washoff[i] = washoff_rate * self.timestep_d
                self.EffectiveWashoff[i] = math.min(self.Washoff[i], buildup_mass)
                buildup_mass -= self.EffectiveWashoff[i]

                bu_time = time_from_buildup.get(self.BuMethod)

            self.BuildUp[i] = buildup_mass
    def Calibrate(self, evaluation,
                    bounds = [
                              [0.2, 0.2], # Thresholdflow/EscMax
                              [0.001, 0.05], # Nw
                              [0.001, 0.05]], # Kw
                    optimization_engine='minimize',
                    x0=[ 0.2, 0.01, 0.01],
                    maxiter=1000,
                    objective_function = 'nse'):
        """Calibrate BuWo model using scipy.minimize on spotpy objective
        functions.

        Args:
            evaluation (array-like): Evaluation values to compare (Bu/Wo)
            bounds (list, optional): Buildup parameters bounds
            x0 (list, optional): Initial condition
            objective_function(any, optional): Objective function, options:
            'nse', 'kge', 'rmse', 'pbias' or custom function.
        """
        
        def objective(p):
            threshold_flow, Nw, Kw = p
            self.threshold_flow=threshold_flow
            self.Nw=Nw
            self.Kw=Kw

            self.Process()
            
            obj_func_dict = {'nse': lambda eval, Washoff: -nashsutcliffe(eval, Washoff),
                             'kge': lambda eval, Washoff: -kge(eval, Washoff),
                             'rmse': rmse,
                             'pbias': pbias}
            
            if type(objective_function) == str:
                return obj_func_dict.get(objective_function)(evaluation, self.Washoff)
            return lambda eval: objective_function(eval, self.Washoff)
        
        if optimization_engine == 'minimize':
            return minimize(objective, x0=x0, bounds=bounds)
        else:
            return differential_evolution(objective, bounds=bounds, maxiter=maxiter)