"""
Muskingum routing method for hydrological flow routing.

This module implements the Muskingum method for hydrological flow routing,
including both linear and non-linear routing methods in upstream and downstream directions.
"""


class Muskingum:
    """
    Implementation of the Muskingum method for hydrological flow routing.

    This class provides methods for flow routing using the Muskingum method,
    including linear and non-linear variants for both upstream and downstream routing.
    """

    @staticmethod
    def downstream_routing(upstream, k, x, dt):
        """
        Perform downstream routing from upstream to downstream using the linear Muskingum method.

        Args:
            upstream (list): List of upstream flow values
            k (float): Muskingum storage coefficient (time)
            x (float): Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
            dt (float): Time step

        Returns:
            list: List of downstream flow values
        """
        # Calculate coefficients
        denominator = (2 * k * (1 - x)) + dt
        c0 = (dt - (2 * k * x)) / denominator
        c1 = (dt + (2 * k * x)) / denominator
        c2 = ((2 * k * (1 - x)) - dt) / denominator

        n = len(upstream)
        downstream = [0] * n

        # Initial downstream value is equal to upstream value
        downstream[0] = upstream[0]

        # Loop through from the second to the last entries
        for i in range(1, n):
            downstream[i] = (
                c0 * upstream[i]
                + c1 * upstream[i - 1]
                + c2 * downstream[i - 1]
            )
        return downstream

    @staticmethod
    def _calculate_rk4_coefficients(
        s_current, k, x, m, dt, inflow_current, inflow_next, is_downstream=True
    ):
        """
        Calculate Runge-Kutta 4th order coefficients for Muskingum routing.
        Args:
            s_current (float): Current storage
            k (float): Muskingum storage coefficient
            x (float): Muskingum weighting factor
            m (float): Muskingum exponent for non-linearity
            dt (float): Time step
            inflow_current (float): Current inflow/outflow value
            inflow_next (float): Next inflow/outflow value
            is_downstream (bool): True for downstream routing, False for upstream
        Returns:
            tuple: The four Runge-Kutta coefficients (k1, k2, k3, k4)
        """
        factor = -1 / (1 - x) if is_downstream else 1 / x
        avg_inflow = 0.5 * (inflow_current + inflow_next)

        k1 = factor * ((s_current / k) ** (1 / m) - inflow_current)
        k2 = factor * (((s_current + 0.5 * dt * k1) / k) ** (1 / m) - avg_inflow)
        k3 = factor * (((s_current + 0.5 * dt * k2) / k) ** (1 / m) - avg_inflow)
        k4 = factor * (((s_current + dt * k3) / k) ** (1 / m) - inflow_next)

        return k1, k2, k3, k4

    @staticmethod
    def downstream_fork(k, x, m, dt, inflow):
        """
        Non-linear Muskingum model with fourth-order Runge-Kutta method for routing
        from upstream to downstream.

        This method implements the non-linear Muskingum model using a fourth-order
        Runge-Kutta numerical integration scheme to route flow from upstream to downstream.

        Args:
            k (float): Muskingum storage coefficient (time)
            x (float): Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
            m (float): Muskingum exponent for non-linearity
            dt (float): Time step
            inflow (list): List of inflow values (upstream hydrograph)

        Returns:
            list: List of outflow values (downstream hydrograph)
        """
        n = len(inflow)
        outflow = [0] * n
        outflow[0] = inflow[0]
        s = [0] * n

        for i in range(n - 1):
            # Calculate storage
            s[i] = k * (x * inflow[i] + (1 - x) * outflow[i]) ** m

            # Calculate Runge-Kutta coefficients
            k1, k2, k3, k4 = Muskingum._calculate_rk4_coefficients(
                s[i], k, x, m, dt, inflow[i], inflow[i + 1], is_downstream=True
            )

            # Update storage and calculate outflow
            s[i + 1] = s[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            outflow[i + 1] = (
                1 / (1 - x) * ((s[i + 1] / k) ** (1 / m) - x * inflow[i + 1])
            )

        return outflow

    @staticmethod
    def upstream_fork(k, x, m, dt, outflow):
        """
        Non-linear Muskingum model with fourth-order Runge-Kutta method for routing
        from downstream to upstream.

        This method implements the non-linear Muskingum model using a fourth-order
        Runge-Kutta numerical integration scheme to route flow from downstream to upstream.

        Args:
            k (float): Muskingum storage coefficient (time)
            x (float): Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
            m (float): Muskingum exponent for non-linearity
            dt (float): Time step
            outflow (list): List of outflow values (downstream hydrograph)

        Returns:
            list: List of inflow values (upstream hydrograph)
        """
        n = len(outflow)
        inflow = [0] * n
        inflow[n - 1] = outflow[n - 1]
        s = [0] * n

        for i in range(n - 1, 0, -1):
            # Calculate storage
            s[i] = k * (x * inflow[i] + (1 - x) * outflow[i]) ** m

            # Calculate Runge-Kutta coefficients
            k1, k2, k3, k4 = Muskingum._calculate_rk4_coefficients(
                s[i], k, x, m, dt, outflow[i], outflow[i - 1], is_downstream=False
            )
            # Update storage and calculate inflow
            s[i - 1] = s[i] - dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            inflow[i - 1] = (
                1 / x * (s[i - 1] / k) ** (1 / m) - (1 - x) / x * outflow[i - 1]
            )

        return inflow
