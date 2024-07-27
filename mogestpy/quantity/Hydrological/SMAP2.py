"""
Soil Moisture Accounting Procedure (SMAP) model alternative implementation
(WIP)
"""
# import numpy as np


class Smap:
    """
    SMAP model
    """

    def __init__(self, Str=100, Crec=0, Capc=40, kkt=30,
                 k2t=.2, Ad=1, Tuin=0, Ebin=0, Ai=2):

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
        if Rsolo > Capc * Str:
            return Crec * Tu * (Rsolo - Capc * Str)
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

    def RunStep(self, prec, etp) -> float:
        self.Tu = self.Tu_calc(self.Rsolo, self.Str)
        Es_ = self.Es_calc(prec, self.Ai, self.Str, self.Rsolo)
        Er = self.Er_calc(prec, etp, Es_, self.Tu)
        Rec = self.Rec_calc(
            self.Crec, self.Tu, self.Rsolo, self.Capc, self.Str)

        self.Rsolo = self.Rsolo_calc(self.Rsolo, prec, Es_, Er, Rec)
        Ed = self.Ed_calc(self.Rsup, self.k2t)
        Eb = self.Eb_calc(self.Rsub, self.kkt)

        self.Rsup = self.Rsup_calc(self.Rsup, Es_, Ed)
        self.Rsub = self.Rsub_calc(self.Rsub, Rec, Eb)

        self.i += 1
        return self.discharge_calc(Ed, Eb, self.Ad)

    def Run(self, prec_arr, etp_arr):
        for prec, etp in zip(prec_arr, etp_arr):
            yield self.RunStep(prec, etp)
