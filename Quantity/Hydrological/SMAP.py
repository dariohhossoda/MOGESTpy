class SMAP:
    """
    Clase SMAP (Soil Moisture Accounting Procedure). Modelo 
    Hidrológico para simulação chuva-vazão em bacias.
    """
    def __init__(self, point, basin):
        self.Point = point
        self.Basin = basin
        self.Q = []
    class Point:
        """
        Representação de Point no modelo SMAP
                
        Atributos
        ----
            P : ponto de controle com a série de precipitações
            (array)
            EP : evapotranspiração potencial
            n : tamanho da série de precipitações
        """
        def __init__(self, P, EP):
            self.P = P
            self.EP = EP
            self.n = len(P)

    class Basin:
        """
        Representação da bacia hidrográfica no modelo SMAP
        
        Atributos
        ----
            AD : área de drenagem (km^2)
            Ai : abstração inicial ( = 2.5, default) (mm)
            Capc : capacidade de campo (%)
            kkt : constante de recessão (dias)
            k2t : constante de recessão para o escoamento
            superficial (dias);
            Crec: recarga subterrânea (%);
            Str : capacidade de saturação (mm);
            EBin: escoamento básico inicial (m3/s);
            TUin: teor de umidade inicial (-).

        """
        def __init__(self, Str, AD, Crec, TUin, EBin, Capc,
                     kkt, k2t, Ai=2.5):
            self.Str = Str
            self.Crec = Crec
            self.Tuin = TUin
            self.Ebin = EBin
            self.AD = AD
            self.Capc = Capc
            self.kkt = kkt
            self.k2t = k2t
            self.Ai = Ai

            self.RSolo = TUin * Str
            self.RSup = 0
            self.RSub = EBin / (1 - (.5 ** (1 / kkt))) \
            / AD * 86.4

    def RunModel(self):
        """
        Roda o modelo SMAP, retornando a vazão no exutório
        através da simulação do fluxo d'água nos processos que
        ocorrem na bacia hidrográfica.
        """
        
        for i in range(self.Point.n):
            TU = self.Basin.RSolo / self.Basin.Str

            ES = (self.Point.P[i] - self.Basin.Ai) ** 2 / (
            self.Point.P[i] - self.Basin.Ai + self.Basin.Str -
            self.Basin.RSolo) if (self.Point.P[i] > self.Basin.Ai) else 0

            ER = self.Point.EP if ((self.Point.P[i] - ES) >
            self.Point.EP) else self.Point.P[i] - ES + (
            (self.Point.EP - self.Point.P[i] + ES) * TU)

            Rec = self.Basin.Crec / 100.0 * TU * (self.Basin.RSolo -
            self.Basin.Capc * self.Basin.Str) if (self.Basin.RSolo >
            (self.Basin.Capc * self.Basin.Str)) else 0

            self.Basin.RSolo += self.Point.P[i] - ES - ER - Rec

            if self.Basin.RSolo > self.Basin.Str:
                ES += self.Basin.RSolo - self.Basin.Str
                self.Basin.RSolo = self.Basin.Str

            self.Basin.RSup += ES
            ED = self.Basin.RSup * (1 - (.5 ** (1 / self.Basin.k2t)))
            self.Basin.RSup -= ED

            EB = self.Basin.RSub * (1 - (.5 ** (1 / self.Basin.kkt)))
            self.Basin.RSub += Rec - EB

            self.Q.append((ED + EB) * self.Basin.AD / 86.4)