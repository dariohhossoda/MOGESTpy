import math
from typing import Callable

import numpy as np


class BuildUpWashoff:
    """
    Class for the BuildUp and Washoff model, based on SWMM (Storm Water Management Model).
    """

    def __init__(
        self,
        Bmax: float,
        Nb: float,
        Kb: float,
        threshold_flow: float,
        Nw: float,
        Kw: float,
        BuMethod: int,
        WoMethod: int,
        timestep_h: float,
        initial_buildup: float,
        area: float,
        area_fraction: float,
        surface_flow: np.ndarray,
        landuse_name: str,
        aggregate: bool = False,
    ):
        """Initialize a Buwo object.

        Args:
            Bmax (float): Maximum buildup capacity.
            Nb (float): Parameter for buildup function.
            Kb (float): Parameter for buildup function.
            threshold_flow (float): Threshold flow for washoff.
            Nw (float): Parameter for washoff function.
            Kw (float): Parameter for washoff function.
            BuMethod (int): Buildup function method.
            WoMethod (int): Washoff function method.
            timestep_h (float): Time step in hours.
            initial_buildup (float): Initial buildup value.
            area (float): Area of the landuse.
            area_fraction (float): Fraction of the area covered by the landuse.
            surface_flow (float): Surface flow value.
            landuse_name (str): Name of the landuse.
            aggregate (bool, optional): If True, aggregate the results. Defaults to False.
        """
        self.LanduseName: str = landuse_name

        self.Bmax: float = Bmax
        self.Nb: float = Nb
        self.Kb: float = Kb
        self.Kw: float = Kw
        self.Nw: float = Nw

        self.timestep_h: float = timestep_h
        self.timestep_d: float = self.timestep_h / 24

        self.Washoff: list[float] = []
        self.BuildUp: list[float] = []
        self.EffectiveWashoff: list[float] = []

        self.Aggregate: bool = aggregate
        self.InitialBuildUp: float = initial_buildup

        self.Area: float = area
        self.AreaFraction: float = area_fraction

        self.SurfaceFlow: np.ndarray = surface_flow
        self.ThresholdFlow: float = threshold_flow

        self.BuMethod: int = BuMethod
        self.WoMethod: int = WoMethod

    # region Time from BuildUp Equations
    def TimeFromBuildUpPow(self, buildup: float) -> float:
        """
        Returns time from buildup using the power equation.
        """

        try:
            time = (buildup / self.Kb) ** (1 / self.Nb)
        except:
            time = 0

        return time

    def TimeFromBuildUpExp(self, buildup: float) -> float:
        """
        Returns time from buildup using the exponential equation.
        """

        try:
            time = -math.log(1 - buildup / self.Bmax) / self.Kb
        except:
            time = 0

        return time

    def TimeFromBuildUpSat(self, buildup: float) -> float:
        """
        Returns time from buildup using the saturation equation.
        """

        try:
            time = buildup * self.Kb / (self.Bmax - buildup)
        except:
            time = 0

        return time

    # endregion Time from BuildUp Equations

    # region BuildUp Equations
    def BuildUpPow(self, time: float) -> float:
        return math.min(self.Bmax, self.Kb * (time**self.Nb))

    def BuildUpExp(self, time: float) -> float:
        return self.Bmax * (1 - math.e ** (-self.Kb * time))

    def BuildUpSat(self, time: float) -> float:
        return self.Bmax * time / (self.Kb + time)

    # endregion BuildUp Equations

    # region Washoff Equations
    def WashoffExp(self, surface_flow: float, buildup_mass: float) -> float:
        return self.Kw * (surface_flow**self.Nw) * buildup_mass

    def WashoffRating(self, surface_flow: float, watershed_area: float) -> float:
        return self.Kw * (1000 * surface_flow * watershed_area) ** self.Nw

    def WashoffEMC(self, surface_flow: float, watershed_area: float) -> float:
        return self.Kw * surface_flow * watershed_area

    # endregion Washoff Equations

    def Process(self, verbose: bool = False) -> None:
        """
        Run the BuildUp and Washoff model, calculating pollutant.
        """
        buildup_mass: float = self.InitialBuildUp / (self.Area * self.AreaFraction)

        time_from_buildup: dict[int, Callable[[float], float]] = {
            1: self.TimeFromBuildUpPow,
            2: self.TimeFromBuildUpExp,
            3: self.TimeFromBuildUpSat,
        }

        bu_time: float = time_from_buildup.get(self.BuMethod)(buildup_mass)

        for i in range(len(self.SurfaceFlow)):
            if self.SurfaceFlow[i] < self.ThresholdFlow:
                bu_time += self.timestep_d
                self.Washoff.append(0)

                buildup_curve: dict[int, Callable[[float], float]] = {
                    1: self.BuildUpPow,
                    2: self.BuildUpExp,
                    3: self.BuildUpSat,
                }

                buildup_specific: float = buildup_curve.get(self.BuMethod)(bu_time)
                buildup_mass = buildup_specific * self.Area * self.AreaFraction
            else:
                washoff_curve: dict[int, Callable[[float, float], float]] = {
                    1: self.WashoffExp,
                    2: self.WashoffRating,
                    3: self.WashoffEMC,
                }

                washoff_rate: float = washoff_curve.get(self.WoMethod)(
                    self.SurfaceFlow[i], buildup_mass
                )

                self.Washoff.append(washoff_rate * self.timestep_d)

                if verbose:
                    print(
                        f"Washoff: {
                          self.Washoff[-1]}\nBuildup: {buildup_mass}"
                    )

                min_value = min(self.Washoff[-1], buildup_mass)

                self.EffectiveWashoff.append(min_value)
                buildup_mass -= self.EffectiveWashoff[-1]

                buildup_specific = buildup_mass / (self.Area * self.AreaFraction)

                bu_time = time_from_buildup.get(self.BuMethod)(buildup_specific)

            self.BuildUp.append(buildup_mass)
