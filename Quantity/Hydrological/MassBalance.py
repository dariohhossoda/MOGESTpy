"""
Equação de balanço de massa em reservatórios
"""

class MassBalance:
    def __init__(self, Qin, Qout, Delta_t, Vin):
        self.Qin = Qin
        self.Qout = Qout
        self.Delta_t = Delta_t
        self.Vin = Vin
        
    def Volume_out(self):
        return self.Vin - (self.Qin 
                           - self.Qout) * self.Delta_t