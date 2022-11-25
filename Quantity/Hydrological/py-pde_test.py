from pde import FieldCollection, PDEBase, UnitGrid


class SaintVenantPDE(PDEBase):
    """Saint-Venant custom PDE"""

    def __init__(self, stimulus=0.5, τ=10, a=0, b=0, bc="auto_periodic_neumann"):
        super().__init__()
        self.bc = bc
        self.stimulus = stimulus
        self.τ = τ
        self.a = a
        self.b = b

    def evolution_rate(self, state, t=0):
        y, Q = state  # membrane potential and recovery variable

        y_t = y.laplace(bc=self.bc) + y - y**3 / 3 - Q + self.stimulus
        Q_t = (y + self.a - self.b * Q) / self.τ

        return FieldCollection([y_t, Q_t])


grid = UnitGrid([32, 32])
state = FieldCollection.scalar_random_uniform(2, grid)

eq = SaintVenantPDE()
result = eq.solve(state, t_range=100, dt=0.01)
result.plot()