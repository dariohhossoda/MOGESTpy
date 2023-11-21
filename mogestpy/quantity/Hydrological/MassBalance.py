"""
Mass balance equation for reservoirs
"""

class MassBalance:
    """
    Class representing the mass balance of a system.
    
    Attributes:
        Qin (list): List of inflow rates.
        Qout (list): List of outflow rates.
        Delta_t (float): Time interval.
        Vin (float): Initial volume.
    """
    def __init__(self, Qin, Qout, Delta_t, Vin):
        self.Qin = Qin
        self.Qout = Qout
        self.Delta_t = Delta_t
        self.Vin = Vin
        
    def Volume_out(self):
        """
        Calculates the volume outflow over time.
        
        Returns:
            list: List of volume values at each time step.
        """
        vol = [self.Vin]
        for i in range(len(self.Qin) - 1):
            vol.append(vol[-1] + (self.Qin[i + 1] - self.Qout[i + 1]) * self.Delta_t)
        return vol