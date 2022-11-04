class ZeroDimensional:
    """
    Classe de Qualidade da Água em Reservatórios - Modelo 0D
    Zero-Dimensional
    """
    def __init__(self, zero_dim_input, zero_dim_params, timestep, sim_count):
        self.Timestep = timestep
        self.SimCount = sim_count
        self.Input = zero_dim_input
        self.Output = ZeroDimensional.ZeroDOutput()
        self.Params = zero_dim_params
    
    class ZeroDInput:
        """
        Inputs do modelo 0D
        Volume
        Vazão de montante
        Vazão de jusante
        Concentração de montante
        Carga de montante
        Área de contato
        """
        def __init__(self, volume, inflow, outflow, concentration_in, q_load, contact_area):
            self.Volume = volume
            self.Inflow = inflow
            self.Outflow = outflow
            self.ConcentrationIn =concentration_in
            self.Load = q_load
            self.ContactArea = contact_area
            
    class ZeroDParameters:
        """
        Parâmetros do modelo 0D
        
        Coecifiente de Reação
        Velocidade de Assentamento
        """
        def __init__(self):
            self.ReactionCoefficient
            self.SettlingVelocity
        
    class ZeroDOutput:
        """
        Output do modelo 0D
        
        Concentração de jusante
        """
        def __init__(self):
            self.ConcentrationOut = 0
        
    
    def SimulateQuality(self):
        self.SimCount = len(self.Input.Volume())
        _in = self.Input
        _out = self.Output
        _param = self.Params
        
        _out.ConcentrationOut[0] = _in.Concentration[0]
        
        for i in range(1, self.SimCount):
            _out.Concentration_out[i] = (_out.Concentration_out[i - 1] + (self.Timestep / _in.Volume[i]) * (_in.Inflow[i] * _in.Concentration_in[i] + _in.Load[i])) \
                (1 + (self.Timestep / _in.Volume[i]) * \
                    (_in.Outflow[i] + _param.ReactionCoefficient * _in.Volume[i] + _param.SettlingVelocity * _in.ContactArea[i] + ((_in.Volume[i] - _in.Volume[i - 1]) / self.Timestep)))