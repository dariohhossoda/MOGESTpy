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
reservoirs. The model consists of three reservoirs: soil, surface, and
subsurface.

    This SMAP implementation uses the following parameters and units:
    - Str (float): Soil Saturation (mm) (default: 100)
    - Crec (float): Recession Coeficient (%) (default: 0)
    - Capc (float): Field Capacity (%) (default: 40)
    - kkt (float): Base flow recession coefficient (d^-1) (default: 30)
    - k2t (float): Surface runoff recession coefficient (d^-1)
    (default: 0.2)
    - Ad (float): Drainage area (km2) (default: 1)
    - Tuin (float): Initial soil moisture content (-) (default: 0)
    - Ebin (float): Initial base flow (mm) (default: 0)
    - Ai (float): Initial Abstraction (default: 2.5)

    Example:

    Initialization of Smap object:
    >>> smap = Smap(
        Ad=1,
        Str=100,
        Crec=0.1,
        Capc=40,
        kkt=30,
        k2t=.2,
        Tuin=0,
        Ebin=0,
        Ai=2.5
    )

    Running a single step of the model:
    >>> smap.RunStep(10, 5)

    Running the model with a list of precipitation and evapotranspiration
    >>> discharge = smap.run_to_list(
        [1.0, 1.0, 2.0, 0.0, 1.0],
        [0.1, 0.1, 0.1, 0.1, 0.1]
        )
    The result will be a list of discharges with the same length as the
    input lists.
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
        Initializes an instance of the Smap class.
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

        Returns:
            dict: A dictionary containing the bounds for the
            following parameters:
            - "Str": (100, 2000) - Saturation (mm)
            - "Crec": (0, 20) - Recharge coefficient (%)
            - "Capc": (30, 50) - Field capacity (%)
            - "kkt": (30, 180) - Base flow recession coefficient (d^-1)
            - "k2t": (0.2, 10) - Surface runoff recession coefficient (d^-1)
            - "Ai": (2, 5) - Initial abstraction (mm)
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
        Checks if the given parameters are within the defined bounds.

        Args:
            params (dict): A dictionary containing parameter names as keys
            and their values.

        Returns:
            bool: True if all parameters are within bounds, False otherwise.
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
        Soil Reservoir Calculation
        """
        return Rsolo + P - Es - Er - Rec

    def Rsup_calc(self, Rsup, Es, Ed) -> float:
        """
        Surface Reservoir Calculation
        """
        return Rsup + Es - Ed

    def Rsub_calc(self, Rsub, Rec, Eb) -> float:
        """
        Subsurface Reservoir Calculation
        """
        return Rsub + Rec - Eb

    def Rsolo0(self, Tuin, Str) -> float:
        """
        Initial Soil Reservoir Calculation
        """
        return Tuin * Str

    def RSub0(self, Ebin, kkt, Ad) -> float:
        """
        Initial Subsurface Reservoir Calculation
        """
        kt = .5 ** (1 / kkt)
        return Ebin / (1 - kt) / Ad * 86.4

    def RSup0(self) -> float:
        """
        Initial Surface Reservoir Calculation
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
        Executes a single step of the hydrological model with the given
        precipitation and evapotranspiration values.

        The model uses the given parameters to calculate the discharge
        based on the SMAP model equations.
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

        Parameters:
        prec_arr (iterable): An iterable of precipitation values.
        etp_arr (iterable): An iterable of evapotranspiration values.
        reset (bool): If True, resets the model state before running.
        Default is True.

        Yields:
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

        Parameters:
        prec_arr (iterable): An iterable of precipitation values.
        etp_arr (iterable): An iterable of evapotranspiration values.
        reset (bool): If True, resets the model state before running.
        Default is True.

        Returns:
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
        Calibration method for the SMAP model using
        Differential Evolution algorithm from scipy.optimize.

        Note that the objective function will be minimized, therefore
        objective functions such as KGE and NSE should be multiplied by -1.

        - prec_arr (iterable): An iterable of precipitation values
        - etp_arr (iterable): An iterable of evapotranspiration values
        - eval_arr (iterable): An iterable of observed values
        - variables (list[str]): A list of variables to be optimized
        - obj_func (callable): A callable function that receives the
        observed and simulated values and returns a float to be minimized.
        If None, the KGE objective function will be used. The objective
        function parameters are (observed, simulated).

        Returns:
        A scipy.optimize.OptimizeResult object.
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
