"""
Muskingum routing method for hydrological flow routing.

This module implements the Muskingum method for hydrological flow routing,
including both linear and non-linear routing methods in upstream and downstream
directions, and the Muskingum-Cunge method with parameters derived from the
reach characteristics.
"""

import math
from typing import Callable, Sequence

import numpy as np


class Muskingum:
    """
    Implementation of the Muskingum method for hydrological flow routing.

    This class provides methods for flow routing using the Muskingum method,
    including linear and non-linear variants for both upstream and downstream routing.
    """

    @staticmethod
    def downstream_routing(
        upstream: Sequence[float], k: float, x: float, dt: float
    ) -> list[float]:
        """
        Perform downstream routing from upstream to downstream using the linear Muskingum method.

        Parameters
        ----------
        upstream : list
            List of upstream flow values
        k : float
            Muskingum storage coefficient (time)
        x : float
            Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
        dt : float
            Time step

        Returns
        -------
        list
            List of downstream flow values
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
                c0 * upstream[i] + c1 * upstream[i - 1] + c2 * downstream[i - 1]
            )
        return downstream

    @staticmethod
    def _calculate_rk4_coefficients(
        s_current: float,
        k: float,
        x: float,
        m: float,
        dt: float,
        inflow_current: float,
        inflow_next: float,
        is_downstream: bool = True,
    ) -> tuple[float, float, float, float]:
        """
        Calculate Runge-Kutta 4th order coefficients for Muskingum routing.

        Parameters
        ----------
        s_current : float
            Current storage
        k : float
            Muskingum storage coefficient
        x : float
            Muskingum weighting factor
        m : float
            Muskingum exponent for non-linearity
        dt : float
            Time step
        inflow_current : float
            Current inflow/outflow value
        inflow_next : float
            Next inflow/outflow value
        is_downstream : bool
            True for downstream routing, False for upstream

        Returns
        -------
        tuple
            The four Runge-Kutta coefficients (k1, k2, k3, k4)
        """
        factor = -1 / (1 - x) if is_downstream else 1 / x
        avg_inflow = 0.5 * (inflow_current + inflow_next)

        k1 = factor * ((s_current / k) ** (1 / m) - inflow_current)
        k2 = factor * (((s_current + 0.5 * dt * k1) / k) ** (1 / m) - avg_inflow)
        k3 = factor * (((s_current + 0.5 * dt * k2) / k) ** (1 / m) - avg_inflow)
        k4 = factor * (((s_current + dt * k3) / k) ** (1 / m) - inflow_next)

        return k1, k2, k3, k4

    @staticmethod
    def downstream_fork(
        k: float, x: float, m: float, dt: float, inflow: Sequence[float]
    ) -> list[float]:
        """
        Non-linear Muskingum model with fourth-order Runge-Kutta method for routing
        from upstream to downstream.

        This method implements the non-linear Muskingum model using a fourth-order
        Runge-Kutta numerical integration scheme to route flow from upstream to downstream.

        Parameters
        ----------
        k : float
            Muskingum storage coefficient (time)
        x : float
            Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
        m : float
            Muskingum exponent for non-linearity
        dt : float
            Time step
        inflow : list
            List of inflow values (upstream hydrograph)

        Returns
        -------
        list
            List of outflow values (downstream hydrograph)
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
    def upstream_fork(
        k: float, x: float, m: float, dt: float, outflow: Sequence[float]
    ) -> list[float]:
        """
        Non-linear Muskingum model with fourth-order Runge-Kutta method for routing
        from downstream to upstream.

        This method implements the non-linear Muskingum model using a fourth-order
        Runge-Kutta numerical integration scheme to route flow from downstream to upstream.

        Parameters
        ----------
        k : float
            Muskingum storage coefficient (time)
        x : float
            Muskingum weighting factor (dimensionless, 0 ≤ x ≤ 0.5)
        m : float
            Muskingum exponent for non-linearity
        dt : float
            Time step
        outflow : list
            List of outflow values (downstream hydrograph)

        Returns
        -------
        list
            List of inflow values (upstream hydrograph)
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


class MuskingumCunge:
    """
    Implementation of the Muskingum-Cunge method for hydrological flow routing.

    The storage coefficient K and the weighting factor X are computed from the
    reach characteristics instead of being calibrated, following the
    formulation of the MGB model (Collischonn et al., 2020): the kinematic
    celerity comes from Manning's equation for a wide rectangular channel, the
    reach is split into sub-reaches by Fread's (1993) criterion and X is
    Cunge's expression, which matches the numerical diffusion of the scheme to
    the physical diffusion of the flood wave. Routing runs on an internal
    sub-step shorter than the model time step, so reaches with travel time
    shorter than the time step are routed instead of passed through.
    """

    @staticmethod
    def celerity(q_ref: float, manning: float, width: float, slope: float) -> float:
        """
        Kinematic wave celerity for a wide rectangular channel (Manning).

        Parameters
        ----------
        q_ref : float
            Reference discharge (m³/s)
        manning : float
            Manning's roughness coefficient (s/m^(1/3))
        width : float
            Channel width (m)
        slope : float
            Channel slope (m/m)

        Returns
        -------
        float
            Celerity c0 = (5/3) Q0^0.4 S^0.3 / (n^0.6 B^0.4) (m/s)
        """
        return 5.0 / 3.0 * q_ref**0.4 * slope**0.3 / (manning**0.6 * width**0.4)

    @staticmethod
    def parameters(
        length: float,
        q_ref: float,
        manning: float,
        width: float,
        slope: float,
        dt: float,
        x_min: float = 0.2,
        x_max: float = 0.5,
    ) -> dict[str, float]:
        """
        Sub-reach discretization and Muskingum-Cunge coefficients of a reach.

        Parameters
        ----------
        length : float
            Reach length (m)
        q_ref : float
            Reference discharge (m³/s)
        manning : float
            Manning's roughness coefficient (s/m^(1/3))
        width : float
            Channel width (m)
        slope : float
            Channel slope (m/m)
        dt : float
            Desired routing time step (s). It is reduced when
            needed to keep the coefficients non-negative.
        x_min : float, optional
            Lower bound of X. Default is 0.2.
        x_max : float, optional
            Upper bound of X. Default is 0.5.

        Returns
        -------
        dict
            ``celerity`` (m/s), ``n_subreaches``, ``dx`` (m), ``k`` (s),
            ``x``, ``dt`` (effective time step, s) and the routing
            coefficients ``c0``, ``c1`` and ``c2`` of
            ``O[t+1] = c0 I[t+1] + c1 I[t] + c2 O[t]``.

        Raises
        ------
        ValueError
            If a reach characteristic is not positive.
        """
        for name, value in (
            ("length", length),
            ("q_ref", q_ref),
            ("manning", manning),
            ("width", width),
            ("slope", slope),
            ("dt", dt),
        ):
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        celerity = MuskingumCunge.celerity(q_ref, manning, width, slope)

        # Ideal sub-reach length (Fread, 1993)
        term = 1.5 * q_ref / (width * slope * dt * celerity**2)
        dx_ideal = celerity * dt / 2.0 * (1.0 + math.sqrt(1.0 + term))
        n_subreaches = max(1, math.ceil(length / dx_ideal))
        dx = length / n_subreaches
        k = dx / celerity

        # Cunge's X: numerical diffusion equal to the physical diffusion
        x = 0.5 * (1.0 - q_ref / (width * slope * celerity * dx))
        x = min(max(x, x_min), x_max)

        # Non-negative coefficients: 2KX <= dt <= 2K(1 - X)
        dt = min(dt, 2.0 * k * (1.0 - x))
        if dt < 2.0 * k * x:
            x = dt / (2.0 * k)

        c0, c1, c2 = MuskingumCunge._coefficients(k, x, dt)

        return {
            "celerity": celerity,
            "n_subreaches": n_subreaches,
            "dx": dx,
            "k": k,
            "x": x,
            "dt": dt,
            "c0": c0,
            "c1": c1,
            "c2": c2,
        }

    @staticmethod
    def _coefficients(k: float, x: float, dt: float) -> tuple[float, float, float]:
        """
        Linear Muskingum routing coefficients.

        Parameters
        ----------
        k : float
            Storage coefficient (s)
        x : float
            Weighting factor
        dt : float
            Time step (s)

        Returns
        -------
        tuple
            (c0, c1, c2) of ``O[t+1] = c0 I[t+1] + c1 I[t] + c2 O[t]``
        """
        denominator = 2.0 * k * (1.0 - x) + dt
        c0 = (dt - 2.0 * k * x) / denominator
        c1 = (dt + 2.0 * k * x) / denominator
        c2 = (2.0 * k * (1.0 - x) - dt) / denominator
        return c0, c1, c2

    @staticmethod
    def downstream_routing(
        upstream: Sequence[float],
        length: float,
        q_ref: float,
        manning: float,
        width: float,
        slope: float,
        dt: float = 86400.0,
        sub_dt: float = 3600.0,
        x_min: float = 0.2,
        x_max: float = 0.5,
    ) -> np.ndarray:
        """
        Route a hydrograph through a reach with the Muskingum-Cunge method.

        The inflow is held constant within each model time step and routed on
        an internal sub-step through a cascade of equal sub-reaches; the
        outflow of each time step is the mean of its sub-steps. The reach
        starts in steady state with the first inflow value.

        Parameters
        ----------
        upstream : list
            Upstream (inflow) discharge per model time step (m³/s)
        length : float
            Reach length (m)
        q_ref : float
            Reference discharge (m³/s), e.g. regionalized as
            ``A^0.7 / 3`` with the drainage area A in km² (MGB)
        manning : float
            Manning's roughness coefficient (s/m^(1/3))
        width : float
            Channel width (m)
        slope : float
            Channel slope (m/m)
        dt : float, optional
            Model time step (s). Default is 86400 (1 day).
        sub_dt : float, optional
            Desired internal time step (s). Default is 3600
            (1 h). It is reduced when needed to keep the coefficients
            non-negative and to divide ``dt`` into equal sub-steps.
        x_min : float, optional
            Lower bound of X. Default is 0.2.
        x_max : float, optional
            Upper bound of X. Default is 0.5.

        Returns
        -------
        numpy.ndarray
            Downstream (outflow) discharge per model time step (m³/s)
        """
        from scipy.signal import lfilter, lfilter_zi

        inflow = np.asarray(upstream, dtype=float)
        if inflow.size == 0:
            return inflow.copy()

        params = MuskingumCunge.parameters(
            length, q_ref, manning, width, slope, min(sub_dt, dt), x_min, x_max
        )

        # Recompute the coefficients with a sub-step that divides dt evenly
        n_substeps = max(1, math.ceil(dt / params["dt"]))
        step = dt / n_substeps
        k = params["k"]
        x = min(params["x"], step / (2.0 * k))
        c0, c1, c2 = MuskingumCunge._coefficients(k, x, step)

        b, a = [c0, c1], [1.0, -c2]
        steady_state = lfilter_zi(b, a)

        flow = np.repeat(inflow, n_substeps)
        for _ in range(params["n_subreaches"]):
            flow, _ = lfilter(b, a, flow, zi=steady_state * flow[0])

        return flow.reshape(-1, n_substeps).mean(axis=1)

    @staticmethod
    def router(
        dt: float = 86400.0,
        sub_dt: float = 3600.0,
        x_min: float = 0.2,
        x_max: float = 0.5,
    ) -> Callable:
        """
        Build a routing function for ``run_network``.

        The returned function reads the reach characteristics from the
        subcatchment row of the parameter table: ``length`` (m), ``q_ref``
        (m³/s), ``manning``, ``width`` (m) and ``slope`` (m/m). ``run_network``
        passes the row of the subcatchment that receives the upstream flow, so
        the reach is the one crossing that subcatchment and ``q_ref`` and
        ``width`` refer to the area draining into it. A reach with zero length
        passes the flow through.

        Parameters
        ----------
        dt : float, optional
            Model time step (s). Default is 86400 (1 day).
        sub_dt : float, optional
            Desired internal time step (s). Default is 3600.
        x_min : float, optional
            Lower bound of X. Default is 0.2.
        x_max : float, optional
            Upper bound of X. Default is 0.5.

        Returns
        -------
        callable
            ``router(q_upstream, row)`` returning the routed discharge
        """

        def route(q_upstream: np.ndarray, row) -> np.ndarray:
            if float(row["length"]) == 0:
                return np.asarray(q_upstream, dtype=float).copy()

            return MuskingumCunge.downstream_routing(
                q_upstream,
                length=float(row["length"]),
                q_ref=float(row["q_ref"]),
                manning=float(row["manning"]),
                width=float(row["width"]),
                slope=float(row["slope"]),
                dt=dt,
                sub_dt=sub_dt,
                x_min=x_min,
                x_max=x_max,
            )

        return route
