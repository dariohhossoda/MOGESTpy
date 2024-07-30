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

        c0 = (T - (2 * K * x)) / ((2 * K * (1 - x)) + T)
        c1 = (T + (2 * K * x)) / ((2 * K * (1 - x)) + T)
        c2 = ((2 * K * (1 - x)) - T) / ((2 * K * (1 - x)) + T)

        n = len(upstream)
        downstream = [0] * n
        # Initial downstream value is equal to upstream value

        downstream[0] = upstream[0]
        # Loop through from the second to the last entries

        for i in range(1, n):
            downstream[i] = (
                c0 * upstream[i] + c1 * upstream[i - 1] + c2 * downstream[i - 1]
            )
        return downstream

    def downstream_fork(K, X, m, T, inflow):
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
        inflow (list): Input hydrograph.

        Returns:
        list: Output hydrograph.
        """
        n = len(inflow)
        outflow = [0] * n
        outflow[0] = inflow[0]
        s = [0] * n
        for i in range(n - 1):
            s[i] = K * (X * inflow[i] + (1 - X) * outflow[i]) ** m
            k1 = -1 / (1 - X) * ((s[i] / K) ** (1 / m) - inflow[i])
            k2 = (
                -1
                / (1 - X)
                * (
                    ((s[i] + 0.5 * T * k1) / K) ** (1 / m)
                    - 0.5 * (inflow[i] + inflow[i + 1])
                )
            )
            k3 = (
                -1
                / (1 - X)
                * (
                    ((s[i] + 0.5 * T * k2) / K) ** (1 / m)
                    - 0.5 * (inflow[i] + inflow[i + 1])
                )
            )
            k4 = -1 / (1 - X) * (((s[i] + 1.0 * T * k3) / K) ** (1 / m) - inflow[i + 1])
            s[i + 1] = s[i] + T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            outflow[i + 1] = (
                1 / (1 - X) * ((s[i + 1] / K) ** (1 / m) - X * inflow[i + 1])
            )
        return outflow

    def upstream_fork(K, X, m, T, outflow):
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
        n = len(outflow)
        inflow = [0] * n
        inflow[n - 1] = outflow[n - 1]
        s = [0] * n
        for i in range(n - 1, 0, -1):
            s[i] = K * (X * inflow[i] + (1 - X) * outflow[i]) ** m
            k1 = 1 / X * ((s[i] / K) ** (1 / m) - outflow[i])
            k2 = (
                1
                / X
                * (
                    ((s[i] + 0.5 * T * k1) / K) ** (1 / m)
                    - 0.5 * (outflow[i] + outflow[i - 1])
                )
            )
            k3 = (
                1
                / X
                * (
                    ((s[i] + 0.5 * T * k2) / K) ** (1 / m)
                    - 0.5 * (outflow[i] + outflow[i - 1])
                )
            )
            k4 = 1 / X * (((s[i] + 1.0 * T * k3) / K)
                          ** (1 / m) - outflow[i - 1])
            s[i - 1] = s[i] - T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            inflow[i - 1] = (
                1 / X * (s[i - 1] / K) ** (1 /
                                           m) - (1 - X) / X * outflow[i - 1]
            )
        return inflow
