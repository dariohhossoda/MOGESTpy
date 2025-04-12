"""
Soil Moisture Accounting Procedure (SMAP) model.
The SMAP model is a lumped rainfall-runoff model based on conceptual
reservoirs.
"""
import warnings

from scipy.optimize import differential_evolution
from spotpy.objectivefunctions import kge


class Smap:
    """
    Soil Moisture Accounting Procedure (SMAP) model.

    The SMAP model is a lumped rainfall-runoff model based on conceptual
    reservoirs. It consists of three reservoirs: soil, surface, and subsurface.

    Attributes
    ----------
    Str : float
        Soil Saturation (mm). Default is 100.
    Crec : float
        Recession Coefficient (%). Default is 0.
    Capc : float
        Field Capacity (%). Default is 40.
    kkt : float
        Base flow recession coefficient (d⁻¹). Default is 30.
    k2t : float
        Surface runoff recession coefficient (d⁻¹). Default is 0.2.
    Ad : float
        Drainage area (km²). Default is 1.
    Tuin : float
        Initial soil moisture content (-). Default is 0.
    Ebin : float
        Initial base flow (mm). Default is 0.
    Ai : float
        Initial abstraction. Default is 2.5.

    Examples
    --------
    Initialize a Smap object:

    >>> smap = Smap(
    ...     Ad=1,
    ...     Str=100,
    ...     Crec=0.1,
    ...     Capc=40,
    ...     kkt=30,
    ...     k2t=0.2,
    ...     Tuin=0,
    ...     Ebin=0,
    ...     Ai=2.5
    ... )

    Run a single step of the model:

    >>> smap.RunStep(10, 5)

    Run the model with a list of precipitation and evapotranspiration:

    >>> discharge = smap.run_to_list(
    ...     [1.0, 1.0, 2.0, 0.0, 1.0],
    ...     [0.1, 0.1, 0.1, 0.1, 0.1]
    ... )

    The result will be a list of discharges with the same length as the input lists.
    """

    def __init__(
        self,
        Str=100.0,
        Crec=0.0,
        Capc=40.0,
        kkt=30.0,
        k2t=0.2,
        Ad=1.0,
        Tuin=0.0,
        Ebin=0.0,
        Ai=2.5
    ) -> None:
        """
        Initialize an instance of the Smap class.

        Parameters
        ----------
        Str : float, optional
            Soil Saturation (mm). Default is 100.0.
        Crec : float, optional
            Recession Coefficient (%). Default is 0.0.
        Capc : float, optional
            Field Capacity (%). Default is 40.0.
        kkt : float, optional
            Base flow recession coefficient (d⁻¹). Default is 30.0.
        k2t : float, optional
            Surface runoff recession coefficient (d⁻¹). Default is 0.2.
        Ad : float, optional
            Drainage area (km²). Default is 1.0.
        Tuin : float, optional
            Initial soil moisture content (-). Default is 0.0.
        Ebin : float, optional
            Initial base flow (mm). Default is 0.0.
        Ai : float, optional
            Initial abstraction. Default is 2.5.
        """
        self.i = 0

        self.Str = Str
        self.Crec = Crec
        self.Capc = Capc

        self.kkt = kkt
        self.k2t = k2t

        self.Ad = Ad

        self.Tuin = Tuin
        self.Ebin = Ebin

        self.Ai = Ai

        self.Rsolo = self.Rsolo0(self.Tuin, self.Str)
        self.Rsub = self.RSub0(self.Ebin, self.kkt, self.Ad)
        self.Rsup = 0

        self.Tu = self.Tu_calc(self.Rsolo, self.Str)
        self.Es = 0
        self.Er = 0
        self.Rec = 0
        self.Ed = 0
        self.Eb = 0

    def __str__(self) -> str:
        """
        Returns a string representation of the Smap object.

        Returns
        -------
        str
            A string that includes the values of all parameters used in
            the model.
        """
        return (
            'Smap Class Object\n'
            f"Str = {self.Str}\n"
            f"Crec = {self.Crec}\n"
            f"Capc = {self.Capc}\n"
            f"Ai = {self.Ai}\n"
            f"kkt = {self.kkt}\n"
            f"k2t = {self.k2t}\n"
            f"Ad = {self.Ad}\n"
            f"Tuin = {self.Tuin}\n"
            f"Ebin = {self.Ebin}"
        )

    def bounds(self) -> dict:
        """
        Returns the bounds for various hydrological parameters.

        Returns
        -------
        dict
            A dictionary containing the bounds for the following parameters:
            - "Str" : tuple of float
            Saturation (mm), range (100, 2000).
            - "Crec" : tuple of float
            Recharge coefficient (%), range (0, 20).
            - "Capc" : tuple of float
            Field capacity (%), range (30, 50).
            - "kkt" : tuple of float
            Base flow recession coefficient (d⁻¹), range (30, 180).
            - "k2t" : tuple of float
            Surface runoff recession coefficient (d⁻¹), range (0.2, 10).
            - "Ai" : tuple of float
            Initial abstraction (mm), range (2, 5).
        """
        return {
            "Str": (100, 2000),
            "Crec": (0, 20),
            "Capc": (30, 50),
            "kkt": (30, 180),
            "k2t": (.2, 10),
            "Ai": (2, 5)
        }

    def check_bounds(self, params: dict) -> bool:
        """
        Parameters
        ----------
        params : dict
            A dictionary containing parameter names as keys and their values.

        Returns
        -------
        bool
            True if all parameters are within bounds, False otherwise.
        """
        for param, value in params.items():
            if param in self.bounds():
                lower, upper = self.bounds()[param]
                if not (lower <= value <= upper):
                    return False
        return True
    # region: Reservoirs Functions

    def Rsolo_calc(self, Rsolo, P, Es, Er, Rec) -> float:
        """
        Soil Reservoir Calculation.

        Parameters
        ----------
        Rsolo : float
            Initial soil reservoir value.
        P : float
            Precipitation input.
        Es : float
            Soil evaporation.
        Er : float
            Runoff or surface water flow.
        Rec : float
            Recharge or infiltration to groundwater.

        Returns
        -------
        float
            Updated soil reservoir value.
        """
        return Rsolo + P - Es - Er - Rec

    def Rsup_calc(self, Rsup, Es, Ed) -> float:
        """
        Surface Reservoir Calculation.

        Parameters
        ----------
        Rsup : float
            Initial surface reservoir value.
        Es : float
            Soil evaporation.
        Ed : float
            Surface runoff.

        Returns
        -------
        float
            Updated surface reservoir value.
        """
        return Rsup + Es - Ed

    def Rsub_calc(self, Rsub, Rec, Eb) -> float:
        """
        Subsurface Reservoir Calculation.

        Parameters
        ----------
        Rsub : float
            Initial subsurface reservoir value.
        Rec : float
            Recharge or infiltration to groundwater.
        Eb : float
            Base flow.

        Returns
        -------
        float
            Updated subsurface reservoir value.
        """
        return Rsub + Rec - Eb

    def Rsolo0(self, Tuin, Str) -> float:
        """
        Calculate the initial soil reservoir value.

        Parameters
        ----------
        Tuin : float
            Initial soil moisture content (-).
        Str : float
            Soil saturation (mm).

        Returns
        -------
        float
            Initial soil reservoir value.
        """
        return Tuin * Str

    def RSub0(self, Ebin, kkt, Ad) -> float:
        """
        Calculate the initial subsurface reservoir value.

        Parameters
        ----------
        Ebin : float
            Initial base flow (mm).
        kkt : float
            Base flow recession coefficient (d⁻¹).
        Ad : float
            Drainage area (km²).

        Returns
        -------
        float
            Initial subsurface reservoir value.
        """
        kt = .5 ** (1 / kkt)
        return Ebin / (1 - kt) / Ad * 86.4

    def RSup0(self) -> float:
        """
        Calculate the initial surface reservoir value.

        Returns
        -------
        float
            The initial value of the surface reservoir, which is set to 0.
        """
        return 0

    # endregion Reservoirs Functions

    # region: Transfer Functions

    def Es_calc(self, P, Ai, Str, Rsolo):
        inf = P - Ai
        if inf > 0:
            return inf ** 2 / (inf + Str - Rsolo)
        return 0

    def Er_calc(self, P, Ep, Es, Tu):
        k = P - Es
        if k > Ep:
            return Ep
        return k + (Ep - k) * Tu

    def Rec_calc(self, Crec, Tu, Rsolo, Capc, Str):
        if Rsolo > Capc / 100 * Str:
            return Crec / 100 * Tu * (Rsolo - Capc / 100 * Str)
        return 0

    def Ed_calc(self, Rsup, k2t):
        k2 = .5 ** (1 / k2t)
        return Rsup * (1 - k2)

    def Eb_calc(self, Rsub, kkt):
        kt = .5 ** (1 / kkt)
        return Rsub * (1 - kt)

    def Tu_calc(self, RSolo, Str):
        return RSolo / Str

    # endregion Transfer Functions

    def discharge_calc(self, Ed, Eb, Ad):
        return (Ed + Eb) * Ad / 86.4

    def run_step(self, prec, etp, reset=False) -> float:
        """
        Executes a single step of the hydrological model using the SMAP model
        equations.

        Parameters
        ----------
        prec : float
            Precipitation value for the current step.
        etp : float
            Evapotranspiration value for the current step.
        reset : bool, optional
            If True, resets the model state to its initial conditions
            (default is False).

        Returns
        -------
        float
            The calculated discharge value for the current step.

        Notes
        -----
        This method updates the internal state of the model, including
        variables such as soil moisture, subsurface flow, and surface flow,
        based on the input precipitation and evapotranspiration values.
        The calculations are performed using the SMAP model equations
        """
        if reset or self.i == 0:
            self.i = 0
            self.Rsolo = self.Rsolo0(self.Tuin, self.Str)
            self.Rsub = self.RSub0(self.Ebin, self.kkt, self.Ad)
            self.Rsup = 0

        self.Tu = self.Tu_calc(self.Rsolo, self.Str)
        self.Es = self.Es_calc(prec, self.Ai, self.Str, self.Rsolo)
        self.Er = self.Er_calc(prec, etp, self.Es, self.Tu)
        self.Rec = self.Rec_calc(
            self.Crec, self.Tu, self.Rsolo, self.Capc, self.Str)

        self.Rsolo = self.Rsolo_calc(
            self.Rsolo, prec, self.Es, self.Er, self.Rec)
        self.Ed = self.Ed_calc(self.Rsup, self.k2t)
        self.Eb = self.Eb_calc(self.Rsub, self.kkt)

        self.Rsup = self.Rsup_calc(self.Rsup, self.Es, self.Ed)
        self.Rsub = self.Rsub_calc(self.Rsub, self.Rec, self.Eb)

        self.i += 1
        return self.discharge_calc(self.Ed, self.Eb, self.Ad)

    def run(self, prec_arr, etp_arr, reset=True):
        """
        Executes the hydrological model with the given precipitation and
        evapotranspiration values as iterables.

        Parameters
        ----------
        prec_arr : iterable
            An iterable of precipitation values.
        etp_arr : iterable
            An iterable of evapotranspiration values.
        reset : bool, optional
            If True, resets the model state before running. Default is True.

        Yields
        ------
        float
            The result of each RunStep call for the given precipitation and
            evapotranspiration values.
        """

        if reset or self.i == 0:
            self.Rsolo = self.Rsolo0(self.Tuin, self.Str)
            self.Rsub = self.RSub0(self.Ebin, self.kkt, self.Ad)
            self.Rsup = 0

        for prec, etp in zip(prec_arr, etp_arr):
            yield self.RunStep(prec, etp)

    def run_to_list(self, prec_arr, etp_arr, reset=True):
        """
        Executes the hydrological model with the given precipitation and
        evapotranspiration values as iterables and returns the results as a
        list.

        Parameters
        ----------
        prec_arr : iterable
            An iterable of precipitation values.
        etp_arr : iterable
            An iterable of evapotranspiration values.
        reset : bool, optional
            If True, resets the model state before running. Default is True.

        Returns
        -------
        list
            A list containing the results of each RunStep call for the given
            precipitation and evapotranspiration values.
        """

        return list(self.Run(prec_arr, etp_arr, reset))

    def calibrate(
        self,
        prec_arr,
        etp_arr,
        eval_arr,
        variables: list[str],
        obj_func=None
    ):
        """
        Calibrate the SMAP model using the Differential Evolution algorithm.

        This method optimizes the specified model parameters to minimize the
        given objective function. Note that the objective function will be
        minimized, so metrics like KGE and NSE should be multiplied by -1.

        Parameters
        ----------
        prec_arr : iterable
            An iterable of precipitation values.
        etp_arr : iterable
            An iterable of evapotranspiration values.
        eval_arr : iterable
            An iterable of observed values.
        variables : list of str
            A list of variable names to be optimized.
        obj_func : callable, optional
            A callable function that takes observed and simulated values as
            input and returns a float to be minimized. If None, the KGE
            objective function will be used. The function signature should be
            `obj_func(observed, simulated)`.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result of the optimization process, as returned by
            `scipy.optimize.differential_evolution`.
        """

        invalid_vars = [var for var in variables if var not in self.__dict__]
        if invalid_vars:
            raise ValueError(
                f"Variables {', '.join(invalid_vars)} do not exist."
            )

        bounds = [self.bounds()[var] for var in variables]

        if obj_func is None:
            def default_obj_func(obs, sim):
                return -kge(obs, sim)
            obj_func = default_obj_func

        def objective(params):
            self.__dict__.update(
                {var: val for var, val in zip(variables, params)}
            )

            return obj_func(eval_arr, list(self.Run(prec_arr, etp_arr)))

        result = differential_evolution(
            func=objective,
            bounds=bounds
        )

        return result

    # Deprecated method names with warnings
    def RunStep(self, *args, **kwargs):
        warnings.warn(
            "RunStep is deprecated and will be removed in a future version. "
            "Use run_step instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.run_step(*args, **kwargs)

    def Run(self, *args, **kwargs):
        warnings.warn(
            "Run is deprecated and will be removed in a future version. "
            "Use run instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.run(*args, **kwargs)

    def RunToList(self, *args, **kwargs):
        warnings.warn(
            "RunToList is deprecated and will be removed in a future version. "
            "Use run_to_list instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.run_to_list(*args, **kwargs)

    def Calibrate(self, *args, **kwargs):
        warnings.warn(
            "Calibrate is deprecated and will be removed in a future version. "
            "Use calibrate instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.calibrate(*args, **kwargs)
