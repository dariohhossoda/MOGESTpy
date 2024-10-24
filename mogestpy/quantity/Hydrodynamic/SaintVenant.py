import math


class TrapezoidalCrossSection:
    """
    Classe de seção transversal trapezoidal.
    """

    def __init__(self, b: float, y: float, m: float):
        self.b = b
        self.y = y
        self.m = m

        self.B = self.TopBase()
        self.Rh = self.WetArea() / self.WetPerimeter()

    def __str__(self):
        return f'Trapezoidal Cross Section:\n\
b: {self.b:.3f}\n\
y: {self.y:.3f}\n\
m: {self.m:.3f}'

    def TopBase(self) -> (float):
        """
        Largura do topo B (m)
        """

        return self.b + 2 * self.m * self.y

    def WetArea(self) -> (float):
        """
        Área molhada
        """

        return self.b * self.y + self.m * self.y ** 2

    def WetPerimeter(self) -> (float):
        """
        Perímetro molhado
        """

        return self.b + 2 * self.y * math.sqrt(1 + self.m ** 2)

    def NormalDepth(self, Q, n, So) -> (float):
        """
        Calcula a profundidade normal pelo método
        de Newton-Raphson
        """

        y_norm = .1
        f = 1
        exp = 2/3
        tolerance = 1e-13
        steps = 0
        max_steps = 100

        area = self.WetArea()
        perimeter = self.WetPerimeter()

        while math.fabs(f) > tolerance and steps < max_steps:
            f = (area ** (1 + exp) / perimeter ** exp
                 - Q * n / So ** .5)
            df = (5 * area ** exp *
                  (self.b + 2 * self.m * y_norm)
                  / perimeter ** exp
                  - 4 * area ** (1 + exp)
                  / perimeter ** (1 + exp)
                  * (1 + self.m ** 2) ** .5 / 3)
            y_norm -= f / df
            steps += 1

        return y_norm

    def AreaDepth(self, area):
        """
        Calcula a profundidade dada uma área molhada
        pelo método de Newton-Raphson.
        """

        y = 1
        step = 0
        f = 1
        tolerance = 1e-13
        max_steps = 100

        while math.fabs(f) > tolerance and step < max_steps:
            f = self.b * y + self.m * y ** 2
            df = self.b + 2 * self.m * y
            y -= (f - area) / df
            step += 1

        return y


class SaintVenant:
    """
    Classe com as equações de Saint-Venant para o routing
    hidrodinâmico unidimensional considerando seção transversal
    trapezoidal.
    """

    def __init__(self, cross_section, Q, n, So, dt, dx, g=9.81):
        self.CrossSection = cross_section

        self.Q = Q
        self.n = n
        self.So = So

        self.dt = dt
        self.dx = dx

        self.g = g

    def Courant(self):
        """
        Retorna o valor do adimensional de Courant
        """

        A = self.CrossSection.WetArea()
        return (self.dt / self.dx
                * (self.g * self.CrossSection.y + self.Q
                   / A) ** .5)

    def CourantCheck(self) -> (bool):
        """
        Retorna bool da condição de estabilidade numérica
        """

        return self.Courant() <= 1

    def QuadraticFroude(self) -> (bool):
        """
        Retorna o número de Froude ao quadrado
        """

        A = self.CrossSection.WetArea()
        return self.Q ** 2 * self.B / (self.g * A ** 3)

    def Sf(self) -> (bool):
        """
        Retorna a declividade da linha de energia
        """

        A = self.CrossSection.WetArea()
        Rh = self.CrossSection.Rh

        return (self.Q * math.fabs(self.Q) * self.n ** 2
                / (A ** 2 * Rh ** (4 / 3)))

    def UpdateValues(self):
        """
        Calcula o estado com base no passo de tempo
        """

        # for i in range(1, self.y ):

        raise NotImplementedError('Ainda não implementado!')

    def RunModel(self):
        """
        Roda o modelo
        """

        raise NotImplementedError('Ainda não implementado!')


def LateralContribution():
    """
    Define a contribuição lateral a ser incluída
    """

    raise NotImplementedError('Ainda não implementado!')


def Average(values_list, index):
    """
    Dado uma lista de valores e um índice, retorna a média centrado
    no índice
    """

    try:
        return .5 * (values_list[index + 1]
                     + values_list[index - 1])
    except Exception:
        return None
