import math


class SaintVenant:
    """
    Classe com as equações de Saint-Venant para o routing
    hidrodinâmico unidimensional considerando seção transversal
    trapezoidal.
    """

    def __init__(self, b, y, m, Q, n, So, dt, dx, g=9.81):
        self.b = b
        self.y = y
        self.m = m

        self.Q = Q
        self.n = n
        self.So = So

        self.dt = dt
        self.dx = dx

        self.g = g

    class TrapezoidalCrossSection:
        """
        Classe de seção transversal trapezoidal.
        """

        def __init__(self, b, m, y):
            self.b = b
            self.m = m
            self.y = y
            self.B = b + 2 * m * y
            self.Rh = self.WetArea() / self.WetPerimeter()

        def WetArea(self):
            """
            Área molhada
            """

            return self.b * self.y + self.m * self.y ** 2

        def WetPerimeter(self):
            """
            Perímetro molhado
            """

            return self.b + 2 * self.y * math.sqrt(1 + self.m ** 2)

        def NormalDepth(self, Q, n, So):
            """
            Calcula a profundidade normal pelo método
            de Newton-Raphson
            """

            # FIXME: Verificar contas
            y_norm = .1
            f = 1
            exp = 2/3
            tolerance = 1e-13
            steps = 0
            max_steps = 100

            while math.fabs(f) > tolerance and steps < max_steps:
                f = (self.WetArea() ** (1 + exp) / self.WetPerimeter ** exp
                     - Q * n / So ** .5)
                df = (5 * self.WetArea() ** exp *
                      (self.b + 2 * self.m * y_norm)
                      / self.WetPerimeter() ** exp
                      - 4 * self.WetArea() ** (1 + exp)
                      / self.WetPerimeter() ** (1 + exp)
                      * (1 + self.m ** 2) ** .5 / 3)
                y_norm -= f / df
                steps += 1

            return y_norm

        def AreaDepth(self, area):
            """
            Calcula a profundidade dada uma área molhada
            pelo método de Newton-Raphson.
            """

            # FIXME: Verificar contas
            y = .1
            step = 0
            f = 1
            tolerance = 1e-13
            max_steps = 100

            while math.fabs(f) > tolerance and step < max_steps:
                f = self.WetArea()
                df = self.b + 2 * self.m * self.y
                y -= (f - area) / df
                step += 1

            return y

    def CourantCheck(self):
        """
        Verificação de estabilidade de Courant
        """

        A = self.TrapezoidalCrossSection.WetArea()
        return (self.dt / self.dx
                * (self.g * self.y + self.Q
                   / A) ** .5)

    def QuadraticFroude(self):
        """
        Retorna o número de Froude ao quadrado
        """

        A = self.TrapezoidalCrossSection.WetArea()
        return self.Q ** 2 * self.B / (self.g * A ** 3)

    def Sf(self):
        """
        Retorna a declividade da linha de energia
        """

        A = self.TrapezoidalCrossSection.WetArea()
        Rh = self.TrapezoidalCrossSection.Rh

        return (self.Q * math.fabs(self.Q) * self.n ** 2
                / (A ** 2 * Rh ** (4 / 3)))

    def RunModel(self):
        raise NotImplementedError('Ainda não implementado!')
