class SMAP:
    def __init__(self):
        raise NotImplementedError('Falta ajustar.')

    class Ponto:
        """
        Ponto de controle com série de precipitações (array)
        e evapotranspiração potencial (fixa)
        """
        def __init__(self, P):
            self.P = P
        EP = 2.4

    class Bacia:
        """
        Representação da bacia hidrográfica contendo área de
        drenagem, capacidade de campo, constante de recessão
        do escoamento básico e abstração inicial (todos fixos)
        """
        def __init__(self, AD, Capc, kkt):
            self.AD = AD
            self.Capc = Capc
            self.kkt = kkt
        Ai = 2.5

    def RunModel(Str, k2t, Crec, TUin, EBin, Ponto, Bacia):
        """
        Roda o modelo SMAP
        ----

        Parâmetros de calibração:
        ---
        Str : capacidade de saturação (mm)

        k2t : constante de recessão para o escoamento superficial (dias)

        Crec: recarga subterrânea (%)

        TUin: teor de umidade inicial (adimensional)

        EBin: escoamento básico inicial (m3/s)
        """
        # Input
        # AD: área de drenagem (km2)
        n, AD = len(Ponto.P), Bacia.AD

        # Inicialização
        Q = []

        # Ai  : abstração inicial (mm)
        # Capc: capacidade de campo (%)
        # kkt : constante de recessão para o escoamento básico (dias)
        Ai, Capc, kkt = Bacia.Ai, Bacia.Capc, Bacia.kkt

        # Reservatórios em t = 0
        RSolo = TUin * Str
        RSup = 0.0
        RSub = EBin / (1 - (0.5 ** (1 / kkt))) / AD * 86.4

        # TODO: documentar variáveis em outro lugar (TU, ES, ER ...)
        for i in range(n):
            TU = RSolo / Str

            ES = (Ponto.P[i] - Ai) ** 2 / (Ponto.P[i] - Ai + Str - RSolo) \
                if (Ponto.P[i] > Ai) else 0

            ER = Ponto.EP if ((Ponto.P[i] - ES) > Ponto.EP) else \
                Ponto.P[i] - ES + ((Ponto.EP - Ponto.P[i] + ES) * TU)

            Rec = Crec / 100.0 * TU * (RSolo - Capc * Str) \
                if (RSolo > (Capc * Str)) else 0

            RSolo += Ponto.P[i] - ES - ER - Rec

            if RSolo > Str:
                ES += RSolo - Str
                RSolo = Str

            RSup += ES
            ED = RSup * (1 - (0.5 ** (1 / k2t)))
            RSup -= ED

            EB = RSub * (1 - (0.5 ** (1 / kkt)))
            RSub += Rec - EB

            Q.append((ED + EB) * Bacia.AD / 86.4)

        return Q
