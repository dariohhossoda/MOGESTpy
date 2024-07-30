"""
Soil Moisture Accounting Procedure (SMAP) model alternative implementation
(WIP)
"""
# import numpy as np


class Smap:
    """
    Soil Moisture Accounting Procedure (SMAP) model.
    The SMAP model is a lumped rainfall-runoff model based on conceptual
reservoirs.
    """

    def __init__(self, Str=100.0, Crec=0.0, Capc=40.0, kkt=30.0,
                 k2t=.2, Ad=1.0, Tuin=0.0, Ebin=0.0, Ai=2.0):
        """
        Initializes an instance of the SMAP2 class.

        Parameters:
        - Str (float): Soil Saturation (mm) (default: 100)
        - Crec (float): Recession Coeficient (default: 0)
        - Capc (float): Field Capacity (default: 40)
        - kkt (float): TODO: Add description (default: 30)
        - k2t (float): TODO: Add description (default: 0.2)
        - Ad (float): Drainage area (km2) (default: 1)
        - Tuin (float): Initial soil moisture content (default: 0)
        - Ebin (float): Initial base flow (default: 0)
        - Ai (float): Initial Abstraction (default: 2)
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
        text = 'Smap Class Object\n'
        text += f"Str: {self.Str}\n"
        text += f"Crec: {self.Crec}\n"
        text += f"Capc: {self.Capc}\n"
        text += f"Ai: {self.Ai}\n"
        text += f"kkt: {self.kkt}\n"
        text += f"k2t: {self.k2t}\n"
        text += f"Ad: {self.Ad}\n"
        text += f"Tuin: {self.Tuin}\n"
        text += f"Ebin: {self.Ebin}"

        return text

    def bounds(self) -> dict:
        return {
            "Str": (100, 2000),
            "Crec": (0, 20),
            "Capc": (30, 50),
            "kkt": (30, 180),
            "k2t": (.2, 10),
            "Ai": (2, 5)
        }
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
        return Tuin * Str

    def RSub0(self, Ebin, kkt, Ad) -> float:
        kt = .5 ** (1 / kkt)
        return Ebin / (1 - kt) / Ad * 86.4

    def RSup0(self) -> float:
        return 0

    # endregion

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

    # endregion

    def discharge_calc(self, Ed, Eb, Ad):
        return (Ed + Eb) * Ad / 86.4

    def RunStep(self, prec, etp, reset=False) -> float:
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

    def Run(self, prec_arr, etp_arr, reset=True):
        if reset:
            self.Rsolo = self.Rsolo0(self.Tuin, self.Str)
            self.Rsub = self.RSub0(self.Ebin, self.kkt, self.Ad)
            self.Rsup = 0

        for prec, etp in zip(prec_arr, etp_arr):
            yield self.RunStep(prec, etp)

    def Calibrate():
        return NotImplementedError('Calibration not implemented yet')
