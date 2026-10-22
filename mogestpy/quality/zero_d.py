class ZeroDimensional:
    """
    Classe de Qualidade da Água em Reservatórios - Modelo 0D
    Zero-Dimensional
    """

    def __init__(self, volume, Qin, Qout, Cin, As, k, v, timestep):
        """
        Initialize the zero-dimensional model.

        Every input series has one value per time step, in consistent units
        (e.g. m³, m³/s, mg/L, m², 1/s, m/s and a time step in seconds).

        Parameters
        ----------
        volume : list of float
            Reservoir volume (V).
        Qin : list of float
            Inflow (Qin).
        Qout : list of float
            Outflow (Qout).
        Cin : list of float
            Concentration at the inflow (Cin).
        As : list of float
            Contact area (As).
        k : list of float
            First-order decay coefficient.
        v : list of float
            Settling velocity.
        timestep : float
            Time step.
        """
        self.Input = ZeroDimensional.Input(volume, Qin, Qout, Cin, As)
        self.SimCount = len(self.Input.Volume)
        self.Output = ZeroDimensional.Output(self.SimCount)
        self.Params = ZeroDimensional.Params(k, v)
        self.Timestep = timestep

    class Output:
        """
        Represents the output of a calculation.

        Attributes
        ----------
        ConcentrationOut : list
            A list of concentration values.
        """

        def __init__(self, size):
            """
            Initialize the output with zero concentrations.

            Parameters
            ----------
            size : int
                Number of time steps.
            """
            self.ConcentrationOut = [0 for _ in range(size)]

    class Input:
        """
        Inputs for the 0D model.

        Attributes
        ----------
        Volume : list of float
            Volume (V).
        Inflow : list of float
            Inflow (Qin).
        Outflow : list of float
            Outflow (Qout).
        ConcentrationIn : list of float
            Concentration at the inflow (Cin).
        ContactArea : list of float
            Contact area (As).
        """

        def __init__(self, volume, inflow, outflow, concentration_in, contact_area):
            """
            Store the input series of the 0D model.

            Parameters
            ----------
            volume : list of float
                Volume (V).
            inflow : list of float
                Inflow (Qin).
            outflow : list of float
                Outflow (Qout).
            concentration_in : list of float
                Concentration at the inflow (Cin).
            contact_area : list of float
                Contact area (As).
            """
            self.Volume = volume
            self.Inflow = inflow
            self.Outflow = outflow
            self.ConcentrationIn = concentration_in
            self.ContactArea = contact_area

    class Params:
        """
        Parameters of the 0D model.

        Attributes
        ----------
        ReactionCoefficient : list of float
            First-order decay coefficient (k).
        SettlingVelocity : list of float
            Settling velocity (v).
        """

        def __init__(self, k, v):
            """
            Store the parameters of the 0D model.

            Parameters
            ----------
            k : list of float
                First-order decay coefficient.
            v : list of float
                Settling velocity.
            """
            self.ReactionCoefficient = k
            self.SettlingVelocity = v

    def RunModel(self):
        r"""
        Run the model over all time steps.

        The concentration starts at zero and each step solves the mass
        balance with an implicit Euler scheme, so it is stable for any time
        step. Results are stored in ``Output.ConcentrationOut``.

        Notes
        -----
        .. math::

           C_i = \frac{C_{i-1} + \frac{\Delta t}{V_i} Q_{in,i} C_{in,i}}
           {1 + \frac{\Delta t}{V_i} \left(Q_{out,i} + k_i V_i + v_i A_{s,i}
           + \frac{V_i - V_{i-1}}{\Delta t}\right)}
        """
        _in = self.Input
        _out = self.Output
        _param = self.Params
        dt = self.Timestep

        _out.ConcentrationOut[0] = 0

        for i in range(1, self.SimCount):
            _out.ConcentrationOut[i] = (
                _out.ConcentrationOut[i - 1]
                + dt / _in.Volume[i] * _in.Inflow[i] * _in.ConcentrationIn[i]
            ) / (
                1
                + dt
                / _in.Volume[i]
                * (
                    _in.Outflow[i]
                    + _param.ReactionCoefficient[i] * _in.Volume[i]
                    + _param.SettlingVelocity[i] * _in.ContactArea[i]
                    + (_in.Volume[i] - _in.Volume[i - 1]) / dt
                )
            )
