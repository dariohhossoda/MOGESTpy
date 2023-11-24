class Muskingum:
    def downstream_routing(upstream, K, x, T):
        """
        Performs downstream routing from upstream to downstream
        using the Muskingum method.

        Parameters:
        - upstream (list): List of upstream values.
        - K (float): Muskingum routing parameter.
        - x (float): Muskingum weighting factor.
        - T (float): Time step.

        Returns:
        - downstream (list): List of downstream values.
        """
        # Coefficients
        C0 = (T - (2 * K * x)) / ((2 * K * (1 - x)) + T)
        C1 = (T + (2 * K * x)) / ((2 * K * (1 - x)) + T)
        C2 = ((2 * K * (1 - x)) - T) / ((2 * K * (1 - x)) + T)

        n = len(upstream)
        downstream = [0] * n
        # Initial downstream value is equal to upstream value
        downstream[0] = upstream[0]
        # Loop through from the second to the last entries
        for i in range(1, n):
            downstream[i] = C0 * upstream[i] + C1 * upstream[i - 1] + C2 * downstream[i - 1]

        return downstream

    def DownstreamFORK(K, X, m, T, I):
        """
        Non-linear Muskingum model (first-order)
        with fourth-order Runge-Kutta method (routing
        from upstream to downstream). Variables K, X, and m
        should be calibrated. I refers to the input, or upstream
        hydrograph, and T to the time step involved (in this case,
        24 hours).

        Parameters:
        K (float): Muskingum storage coefficient.
        X (float): Muskingum weighting factor.
        m (float): Muskingum exponent.
        T (float): Time step.
        I (list): Input hydrograph.

        Returns:
        list: Output hydrograph.
        """
        n = len(I)

        # Outflow
        O = [0] * n
        # Initial value
        O[0] = I[0]
        # Storage
        S = [0] * n

        for i in range(n - 1):
            # Current storage
            S[i] = K * (X * I[i] + (1 - X) * O[i]) ** m
            # Coefficients
            k1 = (-1 / (1 - X)) * ((S[i] / K) ** (1 / m) - I[i])
            k2 = (-1 / (1 - X)) * (((S[i] + 0.5 * T * k1) / K) ** (1 / m) - 0.5 * (I[i] + I[i + 1]))
            k3 = (-1 / (1 - X)) * (((S[i] + 0.5 * T * k2) / K) ** (1 / m) - 0.5 * (I[i] + I[i + 1]))
            k4 = (-1 / (1 - X)) * (((S[i] + 1.0 * T * k3) / K) ** (1 / m) - I[i + 1])
            # Next storage
            S[i + 1] = S[i] + T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Next outflow
            O[i + 1] = (1 / (1 - X)) * ((S[i + 1] / K) ** (1 / m) - X * I[i + 1])

        return O

    def UpstreamFORK(K, X, m, T, O):
        """
        Non-linear Muskingum model (first-order)
        with fourth-order Runge-Kutta method (routing
        from downstream to upstream). Variables K, X, and m
        should be calibrated. O refers to the output, or
        downstream hydrograph, and T refers to the time step
        involved (in this case, 24 hours).

        Parameters:
        K (float): Muskingum storage coefficient
        X (float): Weighting factor for inflow and outflow
        m (float): Non-linearity coefficient
        T (float): Time step
        O (list): List of inflow values

        Returns:
        list: List of upstream inflow values
        """
        n = len(O)

        # Inflow
        I = [0] * n
        # Initial value
        I[n - 1] = O[n - 1]

        # Storage, rate of change
        S = [0] * n

        for i in range(n - 1, 0, -1):
            # Storage
            S[i] = K * (X * I[i] + (1 - X) * O[i]) ** m
            # Coefficients
            k1 = (1 / X) * ((S[i] / K) ** (1 / m) - O[i])
            k2 = (1 / X) * (((S[i] + 0.5 * T * k1) / K) ** (1 / m) - (0.5 * (O[i] + O[i - 1])))
            k3 = (1 / X) * (((S[i] + 0.5 * T * k2) / K) ** (1 / m) - (0.5 * (O[i] + O[i - 1])))
            k4 = (1 / X) * (((S[i] + 1.0 * T * k3) / K) ** (1 / m) - O[i - 1])
            # Storage at t - 1 (previous step)
            S[i - 1] = S[i] - T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Inflow at t - 1 (previous step)
            I[i - 1] = (1 / X) * ((S[i - 1] / K) ** (1 / m)) - ((1 - X) / X) * O[i - 1]

        return I
