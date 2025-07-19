"""
Simulação Hidrodinâmica e de Qualidade da Água (SIHQUAL)
Class-based implementation for hydrodynamic and water quality simulation.
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def _to_numpy_array(data, dtype=None):
    """
    Convert any iterable to numpy array, maintaining pandas DataFrame compatibility.

    Args:
        data: Input data (pandas Series/DataFrame, list, tuple, numpy array, etc.)
        dtype: Optional data type for the array

    Returns:
        numpy array
    """
    if hasattr(data, 'to_numpy'):
        # pandas Series/DataFrame
        return data.to_numpy(dtype=dtype) if dtype else data.to_numpy()
    elif hasattr(data, 'values'):
        # pandas Series/DataFrame (older versions)
        return np.array(data.values, dtype=dtype) if dtype else np.array(data.values)
    else:
        # Any other iterable (list, tuple, numpy array, etc.)
        return np.array(data, dtype=dtype) if dtype else np.array(data)


class SIHQUAL:
    """
    SIHQUAL (Simulação Hidrodinâmica e de Qualidade da Água) class.

    This class implements a 1D hydrodynamic and water quality model for rivers and channels.
    It solves the Saint-Venant equations for hydrodynamics and the advection-dispersion
    equation for water quality.

    Examples:
        Basic usage:

        >>> model = SIHQUAL(dx=100, dt=10, xf=1000, tf=3600)
        >>> model.set_uniform_geometry(
        ...     bottom_width=10.0,
        ...     side_slope=0.0,
        ...     manning_coef=0.03,
        ...     bed_slope=0.001
        ... )
        >>> model.set_uniform_initial_conditions(
        ...     depth=2.0,
        ...     velocity=0.5,
        ...     concentration=0.0
        ... )
        >>> model.set_output_sections([0, 500, 1000])
        >>> results = model.run()
    """

    def __init__(self, dx, dt, xf, tf, alpha=0.6, dispersion_coef=5.0):
        """
        Initialize the SIHQUAL model.

        Args:
            dx: Spatial step (m)
            dt: Time step (s)
            xf: Total length (m)
            tf: Total simulation time (s)
            alpha: Weighting factor for numerical scheme (default: 0.6)
            dispersion_coef: Dispersion coefficient (m²/s) (default: 5.0)
        """
        self.dx = dx
        self.dt = dt
        self.xf = xf
        self.tf = tf
        self.alpha = alpha
        self.dispersion_coef = dispersion_coef

        # Number of spatial points
        self.dim = int(xf / dx) + 1

        # Physical constants
        self.gravity = 9.81

        # Initialize arrays
        self.b1 = None  # Bottom width
        self.m = None   # Side slope
        self.n = None   # Manning's coefficient
        self.So = None  # Bed slope

        self.y1 = None  # Water depth
        self.v1 = None  # Velocity
        self.c1 = None  # Concentration

        self.Kd = None  # Decay coefficient
        self.Ks = None  # Source coefficient

        # Boundary conditions
        self.index_Q = []  # Flow boundary indices
        self.index_y = []  # Water level boundary indices
        self.index_c = []  # Concentration boundary indices

        self.t_arr_Q = None  # Time array for flow boundaries
        self.t_arr_y = None  # Time array for water level boundaries
        self.t_arr_c = None  # Time array for concentration boundaries

        self.bQ_data = None  # Flow boundary data
        self.by_data = None  # Water level boundary data
        self.bc_data = None  # Concentration boundary data

        # Lateral inflows
        self.lQ_slices = []  # Lateral flow segments
        self.lc_slices = []  # Lateral concentration segments

        self.t_arr_lQ = None  # Time array for lateral flow
        self.t_arr_lc = None  # Time array for lateral concentration

        self.lQ_data = None  # Lateral flow data
        self.lc_data = None  # Lateral concentration data

        # Output sections
        self.output_sections = []

        # Auto time step adjustment
        self.auto_step = True  # Enable auto_step by default
        self.courant_max = 0.0

    def set_uniform_geometry(self, bottom_width, side_slope, manning_coef, bed_slope):
        """
        Set uniform channel geometry (constant values along the channel).

        This is a convenience method for setting up a channel with constant
        geometric properties.

        Args:
            bottom_width: Bottom width (m)
            side_slope: Side slope (horizontal:vertical)
            manning_coef: Manning's roughness coefficient
            bed_slope: Bed slope

        Returns:
            self: For method chaining
        """
        self.set_geometry(
            bottom_width=bottom_width,
            side_slope=side_slope,
            manning_coef=manning_coef,
            bed_slope=bed_slope
        )
        return self

    def set_geometry(self, bottom_width, side_slope, manning_coef, bed_slope):
        """
        Set channel geometry.

        Args:
            bottom_width: Bottom width (m) - scalar or array
            side_slope: Side slope (horizontal:vertical) - scalar or array
            manning_coef: Manning's roughness coefficient - scalar or array
            bed_slope: Bed slope - scalar or array

        Returns:
            self: For method chaining
        """
        # Convert scalar inputs to arrays if needed
        if np.isscalar(bottom_width):
            self.b1 = np.ones(self.dim) * bottom_width
        else:
            # Convert to numpy array first
            bottom_width_arr = _to_numpy_array(bottom_width)
            # Interpolate if array length doesn't match dim
            if len(bottom_width_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(bottom_width_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.b1 = np.interp(x_full, x_points, bottom_width_arr)
            else:
                self.b1 = bottom_width_arr

        if np.isscalar(side_slope):
            self.m = np.ones(self.dim) * side_slope
        else:
            # Convert to numpy array first
            side_slope_arr = _to_numpy_array(side_slope)
            # Interpolate if array length doesn't match dim
            if len(side_slope_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(side_slope_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.m = np.interp(x_full, x_points, side_slope_arr)
            else:
                self.m = side_slope_arr

        if np.isscalar(manning_coef):
            self.n = np.ones(self.dim) * manning_coef
        else:
            # Convert to numpy array first
            manning_coef_arr = _to_numpy_array(manning_coef)
            # Interpolate if array length doesn't match dim
            if len(manning_coef_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(manning_coef_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.n = np.interp(x_full, x_points, manning_coef_arr)
            else:
                self.n = manning_coef_arr

        if np.isscalar(bed_slope):
            self.So = np.ones(self.dim) * bed_slope
        else:
            # Convert to numpy array first
            bed_slope_arr = _to_numpy_array(bed_slope)
            # Interpolate if array length doesn't match dim
            if len(bed_slope_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(bed_slope_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.So = np.interp(x_full, x_points, bed_slope_arr)
            else:
                self.So = bed_slope_arr

        return self

    def set_uniform_initial_conditions(self, depth, velocity, concentration=0.0):
        """
        Set uniform initial conditions (constant values along the channel).

        This is a convenience method for setting up initial conditions with
        constant values.

        Args:
            depth: Initial water depth (m)
            velocity: Initial velocity (m/s)
            concentration: Initial concentration (mg/L), default is 0.0

        Returns:
            self: For method chaining
        """
        self.set_initial_conditions(
            initial_depth=depth,
            initial_velocity=velocity,
            initial_concentration=concentration
        )
        return self

    def set_initial_conditions(self, initial_depth, initial_velocity, initial_concentration):
        """
        Set initial conditions.

        Args:
            initial_depth: Initial water depth (m) - scalar or array
            initial_velocity: Initial velocity (m/s) - scalar or array
            initial_concentration: Initial concentration (mg/L) - scalar or array

        Returns:
            self: For method chaining
        """
        # Convert scalar inputs to arrays if needed
        if np.isscalar(initial_depth):
            self.y1 = np.ones(self.dim) * initial_depth
        else:
            # Convert to numpy array first
            initial_depth_arr = _to_numpy_array(initial_depth)
            # Interpolate if array length doesn't match dim
            if len(initial_depth_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(initial_depth_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.y1 = np.interp(x_full, x_points, initial_depth_arr)
            else:
                self.y1 = initial_depth_arr

        if np.isscalar(initial_velocity):
            self.v1 = np.ones(self.dim) * initial_velocity
        else:
            # Convert to numpy array first
            initial_velocity_arr = _to_numpy_array(initial_velocity)
            # Interpolate if array length doesn't match dim
            if len(initial_velocity_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(initial_velocity_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.v1 = np.interp(x_full, x_points, initial_velocity_arr)
            else:
                self.v1 = initial_velocity_arr

        if np.isscalar(initial_concentration):
            self.c1 = np.ones(self.dim) * initial_concentration
        else:
            # Convert to numpy array first
            initial_concentration_arr = _to_numpy_array(initial_concentration)
            # Interpolate if array length doesn't match dim
            if len(initial_concentration_arr) != self.dim:
                x_points = np.linspace(
                    0, self.xf, len(initial_concentration_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.c1 = np.interp(
                    x_full, x_points, initial_concentration_arr)
            else:
                self.c1 = initial_concentration_arr

        return self

    def set_uniform_reaction_parameters(self, decay_coef=0.0, source_coef=0.0):
        """
        Set uniform reaction parameters (constant values along the channel).

        This is a convenience method for setting up reaction parameters with
        constant values.

        Args:
            decay_coef: Decay coefficient (1/s), default is 0.0
            source_coef: Source coefficient (mg/L/s), default is 0.0

        Returns:
            self: For method chaining
        """
        self.set_reaction_parameters(
            decay_coef=decay_coef,
            source_coef=source_coef
        )
        return self

    def set_reaction_parameters(self, decay_coef, source_coef):
        """
        Set reaction parameters for water quality.

        Args:
            decay_coef: Decay coefficient (1/s) - scalar or array
            source_coef: Source coefficient (mg/L/s) - scalar or array

        Returns:
            self: For method chaining
        """
        # Convert scalar inputs to arrays if needed
        if np.isscalar(decay_coef):
            self.Kd = np.ones(self.dim) * decay_coef
        else:
            # Convert to numpy array first
            decay_coef_arr = _to_numpy_array(decay_coef)
            # Interpolate if array length doesn't match dim
            if len(decay_coef_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(decay_coef_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.Kd = np.interp(x_full, x_points, decay_coef_arr)
            else:
                self.Kd = decay_coef_arr

        if np.isscalar(source_coef):
            self.Ks = np.ones(self.dim) * source_coef
        else:
            # Convert to numpy array first
            source_coef_arr = _to_numpy_array(source_coef)
            # Interpolate if array length doesn't match dim
            if len(source_coef_arr) != self.dim:
                x_points = np.linspace(0, self.xf, len(source_coef_arr))
                x_full = np.linspace(0, self.xf, self.dim)
                self.Ks = np.interp(x_full, x_points, source_coef_arr)
            else:
                self.Ks = source_coef_arr

        return self

    def set_output_sections(self, positions):
        """
        Set output sections.

        Args:
            positions: List of positions (m) for output sections

        Returns:
            self: For method chaining
        """
        # Convert positions to indices
        self.output_sections = [self._x_to_index(pos) for pos in positions]
        return self

    def set_evenly_spaced_output_sections(self, num_sections):
        """
        Set evenly spaced output sections along the channel.

        This is a convenience method for setting up output sections at
        evenly spaced intervals.

        Args:
            num_sections: Number of output sections (including endpoints)

        Returns:
            self: For method chaining
        """
        positions = np.linspace(0, self.xf, num_sections)
        return self.set_output_sections(positions)

    def set_boundary_conditions(self, boundary_positions, boundary_data):
        """
        Set boundary conditions.

        Args:
            boundary_positions: Dictionary with positions for each boundary type
                {'Q': [positions], 'y': [positions], 'c': [positions]}
            boundary_data: Dictionary with time series data for each boundary type
                {'Q': {'time': array, 'values': array}, 
                 'y': {'time': array, 'values': array},
                 'c': {'time': array, 'values': array}}

        Returns:
            self: For method chaining
        """
        # Flow boundaries
        if 'Q' in boundary_positions and 'Q' in boundary_data:
            self.index_Q = [self._x_to_index(pos)
                            for pos in boundary_positions['Q']]
            self.t_arr_Q = _to_numpy_array(boundary_data['Q']['time'])
            self.bQ_data = _to_numpy_array(boundary_data['Q']['values'])

        # Water level boundaries
        if 'y' in boundary_positions and 'y' in boundary_data:
            self.index_y = [self._x_to_index(pos)
                            for pos in boundary_positions['y']]
            self.t_arr_y = _to_numpy_array(boundary_data['y']['time'])
            self.by_data = _to_numpy_array(boundary_data['y']['values'])

        # Concentration boundaries
        if 'c' in boundary_positions and 'c' in boundary_data:
            self.index_c = [self._x_to_index(pos)
                            for pos in boundary_positions['c']]
            self.t_arr_c = _to_numpy_array(boundary_data['c']['time'])
            self.bc_data = _to_numpy_array(boundary_data['c']['values'])

        return self

    def set_simple_upstream_flow(self, flow_values, times=None):
        """
        Set a simple upstream flow boundary condition.

        This is a convenience method for setting up a flow boundary condition
        at the upstream end of the channel.

        Args:
            flow_values: Flow values (m³/s) - scalar or array
            times: Time values (s) - array, if None, uses [0, tf]

        Returns:
            self: For method chaining
        """
        if times is None:
            times = np.array([0, self.tf])
        else:
            times = _to_numpy_array(times)

        if np.isscalar(flow_values):
            flow_values = np.array([[flow_values] * len(times)])
        else:
            flow_values_arr = _to_numpy_array(flow_values)
            # Ensure flow_values and times have the same length
            if len(flow_values_arr) != len(times):
                raise ValueError(
                    f"flow_values length ({len(flow_values_arr)}) must match times length ({len(times)})")
            flow_values = np.array([flow_values_arr])

        boundary_positions = {'Q': [0.0]}
        boundary_data = {'Q': {'time': times, 'values': flow_values}}

        return self.set_boundary_conditions(boundary_positions, boundary_data)

    def set_simple_downstream_level(self, level_values, times=None):
        """
        Set a simple downstream water level boundary condition.

        This is a convenience method for setting up a water level boundary condition
        at the downstream end of the channel.

        Args:
            level_values: Water level values (m) - scalar or array
            times: Time values (s) - array, if None, uses [0, tf]

        Returns:
            self: For method chaining
        """
        if times is None:
            times = np.array([0, self.tf])
        else:
            times = _to_numpy_array(times)

        if np.isscalar(level_values):
            level_values = np.array([[level_values] * len(times)])
        else:
            level_values_arr = _to_numpy_array(level_values)
            # Ensure level_values and times have the same length
            if len(level_values_arr) != len(times):
                raise ValueError(
                    f"level_values length ({len(level_values_arr)}) must match times length ({len(times)})")
            level_values = np.array([level_values_arr])

        boundary_positions = {'y': [self.xf]}
        boundary_data = {'y': {'time': times, 'values': level_values}}

        return self.set_boundary_conditions(boundary_positions, boundary_data)

    def set_lateral_inflows(self, lateral_Q_segments=None, lateral_Q_data=None,
                            lateral_c_segments=None, lateral_c_data=None):
        """
        Set lateral inflows.

        Args:
            lateral_Q_segments: List of tuples with start and end positions (m) for lateral flow
            lateral_Q_data: Dictionary with time series data for lateral flow
                {'time': array, 'values': array}
            lateral_c_segments: List of tuples with start and end positions (m) for lateral concentration
            lateral_c_data: Dictionary with time series data for lateral concentration
                {'time': array, 'values': array}

        Returns:
            self: For method chaining
        """
        # Lateral flow
        if lateral_Q_segments is not None and lateral_Q_data is not None:
            self.lQ_slices = [(self._x_to_index(start), self._x_to_index(end))
                              for start, end in lateral_Q_segments]
            self.t_arr_lQ = _to_numpy_array(lateral_Q_data['time'])
            self.lQ_data = _to_numpy_array(lateral_Q_data['values'])

        # Lateral concentration
        if lateral_c_segments is not None and lateral_c_data is not None:
            self.lc_slices = [(self._x_to_index(start), self._x_to_index(end))
                              for start, end in lateral_c_segments]
            self.t_arr_lc = _to_numpy_array(lateral_c_data['time'])
            self.lc_data = _to_numpy_array(lateral_c_data['values'])

        return self

    def enable_auto_step_adjustment(self, enable=True):
        """
        Enable or disable automatic time step adjustment.

        Args:
            enable: Whether to enable auto step adjustment (default: True)

        Returns:
            self: For method chaining
        """
        self.auto_step = enable
        return self

    def run(self, show_progress=True):
        """
        Run the simulation.

        Args:
            show_progress: Whether to show a progress bar (default: True)

        Returns:
            DataFrame with simulation results
        """
        # Check if all required parameters are set
        self._check_parameters()

        # Initialize arrays for the next time step
        y2 = np.zeros_like(self.y1)
        v2 = np.zeros_like(self.v1)
        c2 = np.zeros_like(self.c1)

        # Initialize lateral inflow arrays
        ql = np.zeros(self.dim)
        cqd = np.zeros(self.dim)

        # Initialize progress bar
        if show_progress:
            progress = tqdm(total=self.tf, desc='SIHQUAL', unit='s_(sim)')

        # Initialize output variables
        n_index = 0
        sim_time = 0

        days_total = int(self.tf // 86400) + 1
        variables = ['Q', 'y', 'c']
        cols = pd.MultiIndex.from_product(
            [self.output_sections, variables], names=['section', 'variables'])
        t_index = pd.Index(list(range(days_total)), name='t')

        col_num = len(variables) * len(self.output_sections)
        df_data = np.zeros((days_total, col_num))

        # Main simulation loop
        while sim_time <= self.tf:
            # Calculate hydraulic parameters
            A1 = self._wet_area(self.b1, self.y1, self.m)
            B1 = self._top_width(self.b1, self.y1, self.m)
            Rh1 = self._hydraulic_radius(self.b1, self.y1, self.m)
            Sf1 = self._friction_slope(self.n, self.v1, Rh1)

            # Check stability and adjust time step if needed
            if self.auto_step and self.courant_max > 0:
                self.dt = 0.5 * self.dt / self.courant_max

            # Update lateral inflows
            for i, (start, end) in enumerate(self.lQ_slices):
                if self.t_arr_lQ is not None and self.lQ_data is not None:
                    ql[slice(start, end)] = np.interp(
                        sim_time + self.dt, self.t_arr_lQ, self.lQ_data[i])

            for i, (start, end) in enumerate(self.lc_slices):
                if self.t_arr_lc is not None and self.lc_data is not None:
                    cqd[slice(start, end)] = np.interp(
                        sim_time + self.dt, self.t_arr_lc, self.lc_data[i])

            # Avoid division by zero
            mask = A1 > 0
            cq = np.zeros_like(cqd)
            cq[mask] = cqd[mask] / A1[mask]

            # Hydrodynamic module
            yy, dy = self._avg(self.y1), self._cdelta(self.y1)
            vv, dv = self._avg(self.v1), self._cdelta(self.v1)
            AA, dA = self._avg(A1), self._cdelta(A1)
            BB = self._avg(B1)
            SSf = self._avg(Sf1)

            mfactor = -0.5 * self.dt / self.dx
            y2[1:-1] = (self.alpha * self.y1[1:-1] + (1 - self.alpha) * yy
                        + mfactor * vv * dy
                        + mfactor * vv * dA / BB
                        + mfactor * AA * dv / BB
                        + ql[1:-1] * self.dt / BB)

            v2[1:-1] = (self.alpha * self.v1[1:-1] + (1 - self.alpha) * vv
                        + mfactor * vv * dv
                        + mfactor * self.gravity * dy
                        + self.gravity * self.dt * (self.So[1:-1] - SSf))

            y2[-1] = y2[-2]
            v2[-1] = v2[-2]

            # Water quality module
            dc = self._cdelta(self.c1)

            # Avoid division by zero in water quality calculations
            mask = A1[1:-1] > 0
            c2[1:-1] = self.c1[1:-1]  # Default: keep previous concentration

            # Only update concentration where we have water
            if np.any(mask):
                c2_update = (self.c1[1:-1][mask]
                             - 0.5 * self.dt / self.dx *
                             self.v1[1:-1][mask] * dc[mask]
                             + self.dispersion_coef * 0.5 *
                             self.dt / self.dx / A1[1:-1][mask]
                             * dA[mask] * dc[mask] * 0.5 / self.dx
                             + self.dispersion_coef * self.dt / self.dx ** 2
                             - self.Kd[1:-1][mask] *
                             self.c1[1:-1][mask] * self.dt
                             + self.Ks[1:-1][mask] * self.dt)
                c2[1:-1][mask] = c2_update

            c2[-1] = c2[-2]

            # Apply boundary conditions
            for i, idx in enumerate(self.index_y):
                if self.t_arr_y is not None and self.by_data is not None:
                    y2[idx] = np.interp(sim_time + self.dt,
                                        self.t_arr_y, self.by_data[i])

            A2 = self._wet_area(self.b1, y2, self.m)

            for i, idx in enumerate(self.index_Q):
                if self.t_arr_Q is not None and self.bQ_data is not None:
                    if A2[idx] > 0:
                        v2[idx] = np.interp(
                            sim_time + self.dt, self.t_arr_Q, self.bQ_data[i]) / A2[idx]

            for i, idx in enumerate(self.index_c):
                if self.t_arr_c is not None and self.bc_data is not None:
                    c2[idx] = np.interp(sim_time + self.dt,
                                        self.t_arr_c, self.bc_data[i])

            # Update variables for next time step
            self.y1 = np.copy(y2)
            self.v1 = np.copy(v2)
            self.c1 = np.copy(c2)

            # Check stability
            courant_values = self._courant_number(
                self.dt, self.dx, self.v1, self.gravity, A1, B1)
            # Filter out NaN and inf values that might occur with zero depth
            valid_courant = courant_values[np.isfinite(courant_values)]
            if len(valid_courant) > 0:
                self.courant_max = valid_courant.max()
                if self.courant_max >= 1 and not self.auto_step:
                    raise ValueError(
                        f'Stability error! Courant number = {self.courant_max:.4f} >= 1.0')

            # Store output
            days_elapsed = sim_time / 86400
            if days_elapsed >= n_index:
                k = 0
                _array = np.zeros(col_num)
                for section in self.output_sections:
                    # Ensure positive flow values for output
                    flow = max(0.0, self.v1[section] * A1[section])  # Flow
                    _array[k] = flow
                    _array[k+1] = self.y1[section]              # Depth
                    _array[k+2] = self.c1[section]              # Concentration
                    k += 3
                df_data[int(days_elapsed)] = _array

                n_index += 1

            # Update simulation time
            sim_time += self.dt

            # Update progress bar
            if show_progress:
                progress.update(n=self.dt)

        # Close progress bar
        if show_progress:
            progress.close()

        # Create output DataFrame
        output_df = pd.DataFrame(df_data, columns=cols, index=t_index)
        return output_df

    def save_results(self, results, filename):
        """
        Save simulation results to Excel file.

        Args:
            results: DataFrame with simulation results
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        try:
            results.to_excel(filename)
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

    def plot_results(self, results, variable='Q', figsize=(10, 6), dpi=100):
        """
        Plot simulation results.

        Args:
            results: DataFrame with simulation results
            variable: Variable to plot ('Q', 'y', or 'c')
            figsize: Figure size (width, height) in inches
            dpi: DPI for the figure

        Returns:
            Figure and axes objects
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get sections
        sections = results.columns.get_level_values('section').unique()

        # Plot variable at each section
        for section in sections:
            ax.plot(results.index, results[(section, variable)],
                    label=f'Section {section}')

        # Set labels and title
        variable_labels = {
            'Q': 'Flow (m³/s)',
            'y': 'Water Depth (m)',
            'c': 'Concentration (mg/L)'
        }

        variable_titles = {
            'Q': 'Flow at Different Sections',
            'y': 'Water Depth at Different Sections',
            'c': 'Concentration at Different Sections'
        }

        ax.set_title(variable_titles.get(
            variable, f'{variable} at Different Sections'))
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(variable_labels.get(variable, variable))
        ax.grid(True)
        ax.legend()

        return fig, ax

    def _check_parameters(self):
        """Check if all required parameters are set."""
        if self.b1 is None or self.m is None or self.n is None or self.So is None:
            raise ValueError("Geometry not set. Call set_geometry() first.")

        if self.y1 is None or self.v1 is None or self.c1 is None:
            raise ValueError(
                "Initial conditions not set. Call set_initial_conditions() first.")

        if self.Kd is None or self.Ks is None:
            # Set default values if not provided
            self.Kd = np.zeros(self.dim)
            self.Ks = np.zeros(self.dim)

        if not self.output_sections:
            # Set default output sections if not provided
            self.output_sections = [0, self.dim - 1]

    def _x_to_index(self, x):
        """
        Convert position to array index.

        Args:
            x: Position (m)

        Returns:
            Index in the array
        """
        return int(x / self.xf * (self.dim - 1))

    def _avg(self, vector):
        """
        Calculate average of adjacent elements.

        Args:
            vector: Input array

        Returns:
            Array with averages
        """
        return 0.5 * (vector[2:] + vector[:-2])

    def _cdelta(self, vector):
        """
        Calculate central difference.

        Args:
            vector: Input array

        Returns:
            Array with central differences
        """
        return vector[2:] - vector[:-2]

    def _top_width(self, b, y, m):
        """
        Calculate top width of the channel.

        Args:
            b: Bottom width (m)
            y: Water depth (m)
            m: Side slope

        Returns:
            Top width (m)
        """
        return b + 2 * m * y

    def _wet_area(self, b, y, m):
        """
        Calculate wet area of the channel.

        Args:
            b: Bottom width (m)
            y: Water depth (m)
            m: Side slope

        Returns:
            Wet area (m²)
        """
        return b * y + m * y * y

    def _wet_perimeter(self, b, y, m):
        """
        Calculate wet perimeter of the channel.

        Args:
            b: Bottom width (m)
            y: Water depth (m)
            m: Side slope

        Returns:
            Wet perimeter (m)
        """
        return b + 2 * y * np.sqrt(1 + m * m)

    def _hydraulic_radius(self, b, y, m):
        """
        Calculate hydraulic radius of the channel.

        Args:
            b: Bottom width (m)
            y: Water depth (m)
            m: Side slope

        Returns:
            Hydraulic radius (m)
        """
        return self._wet_area(b, y, m) / self._wet_perimeter(b, y, m)

    def _friction_slope(self, n, v, rh):
        """
        Calculate friction slope.

        Args:
            n: Manning's coefficient
            v: Velocity (m/s)
            rh: Hydraulic radius (m)

        Returns:
            Friction slope
        """
        # Avoid division by zero
        mask = rh > 0
        result = np.zeros_like(rh)
        result[mask] = n[mask] * n[mask] * \
            v[mask] * v[mask] * rh[mask] ** (-4/3)
        return result

    def _courant_number(self, dt, dx, v, g, A, B):
        """
        Calculate Courant number.

        Args:
            dt: Time step (s)
            dx: Spatial step (m)
            v: Velocity (m/s)
            g: Gravity acceleration (m/s²)
            A: Wet area (m²)
            B: Top width (m)

        Returns:
            Courant number
        """
        # Avoid division by zero and negative values
        mask = (B > 0) & (A > 0)
        result = np.zeros_like(v)
        result[mask] = dt / dx * \
            (np.abs(v[mask]) + np.sqrt(g * A[mask] / B[mask]))
        return result

    @classmethod
    def create_simple_channel(cls, length, dx=100.0, dt=10.0, simulation_time=3600.0,
                              bottom_width=10.0, side_slope=0.0, manning=0.03, slope=0.001,
                              depth=2.0, velocity=0.5, concentration=0.0):
        """
        Create a simple rectangular channel with uniform properties.

        This is a convenience method for quickly setting up a simple channel
        with uniform properties.

        Args:
            length: Channel length (m)
            dx: Spatial step (m), default is 100.0
            dt: Time step (s), default is 10.0
            simulation_time: Total simulation time (s), default is 3600.0
            bottom_width: Bottom width (m), default is 10.0
            side_slope: Side slope (horizontal:vertical), default is 0.0
            manning: Manning's roughness coefficient, default is 0.03
            slope: Bed slope, default is 0.001
            depth: Initial water depth (m), default is 2.0
            velocity: Initial velocity (m/s), default is 0.5
            concentration: Initial concentration (mg/L), default is 0.0

        Returns:
            SIHQUAL model instance
        """
        # Create model
        model = cls(dx=dx, dt=dt, xf=length, tf=simulation_time)

        # Set geometry
        model.set_uniform_geometry(
            bottom_width=bottom_width,
            side_slope=side_slope,
            manning_coef=manning,
            bed_slope=slope
        )

        # Set initial conditions
        model.set_uniform_initial_conditions(
            depth=depth,
            velocity=velocity,
            concentration=concentration
        )

        # Set default output sections
        model.set_evenly_spaced_output_sections(5)

        return model

    @classmethod
    def from_excel(cls, filename):
        """
        Create a SIHQUAL model from an Excel file.

        Args:
            filename: Path to Excel file

        Returns:
            SIHQUAL model instance
        """
        try:
            # Read data from Excel
            data_df = pd.read_excel(filename, sheet_name='Data')
            param_df = pd.read_excel(filename, sheet_name='Parameters')

            # Get parameters
            dx = param_df['dx'][0]
            dt = param_df['dt'][0]
            xf = param_df['xf'][0]
            tf = param_df['tf'][0]
            alpha = param_df['alpha'][0]
            dispersion_coef = param_df['D'][0]

            # Create model
            model = cls(dx=dx, dt=dt, xf=xf, tf=tf, alpha=alpha,
                        dispersion_coef=dispersion_coef)

            # Get geometry
            b1 = _to_numpy_array(data_df['b1'], dtype=np.float64)
            m = _to_numpy_array(data_df['m'], dtype=np.float64)
            n = _to_numpy_array(data_df['n'], dtype=np.float64)
            So = _to_numpy_array(data_df['So'], dtype=np.float64)

            # Set geometry
            model.set_geometry(b1, m, n, So)

            # Get initial conditions
            y1 = _to_numpy_array(data_df['y1'], dtype=np.float64)
            v1 = _to_numpy_array(data_df['v1'], dtype=np.float64)
            c1 = _to_numpy_array(data_df['c1'], dtype=np.float64)

            # Set initial conditions
            model.set_initial_conditions(y1, v1, c1)

            # Get reaction parameters
            Kd = _to_numpy_array(data_df['kd'], dtype=np.float64)
            Ks = _to_numpy_array(data_df['ks'], dtype=np.float64)

            # Set reaction parameters
            model.set_reaction_parameters(Kd, Ks)

            # Get output sections
            if 'output_sections' in param_df.columns:
                sections_data = param_df['output_sections'][0]
                # Handle case where output_sections is stored as a string representation of a list
                if isinstance(sections_data, str):
                    # Try to evaluate the string as a Python literal
                    try:
                        import ast
                        sections = ast.literal_eval(sections_data)
                    except (ValueError, SyntaxError):
                        # If that fails, try to parse as comma-separated values
                        sections = [float(x.strip())
                                    for x in sections_data.strip('[]').split(',')]
                else:
                    # If it's already a list or array, use it directly
                    sections = sections_data
                model.set_output_sections(sections)
            else:
                # Set default output sections if not provided
                model.set_output_sections([0, model.xf])

            # Try to load boundary conditions
            input_folder = os.path.dirname(filename)
            # Check if SIHQUALInputs directory exists in the same folder as the Excel file
            sihqual_inputs_dir = os.path.join(input_folder, 'SIHQUALInputs')
            if os.path.isdir(sihqual_inputs_dir):
                input_folder = sihqual_inputs_dir

            try:
                # Load boundary conditions
                boundary_Q = pd.read_csv(os.path.join(input_folder, 'boundary_Q.csv'),
                                         sep=';', encoding='utf-8')
                boundary_y = pd.read_csv(os.path.join(input_folder, 'boundary_y.csv'),
                                         sep=';', encoding='utf-8')
                boundary_c = pd.read_csv(os.path.join(input_folder, 'boundary_c.csv'),
                                         sep=';', encoding='utf-8')

                # Process boundary conditions
                t_arr_Q = _to_numpy_array(
                    boundary_Q.iloc[:, 0], dtype=np.int32)
                t_arr_y = _to_numpy_array(
                    boundary_y.iloc[:, 0], dtype=np.int32)
                t_arr_c = _to_numpy_array(
                    boundary_c.iloc[:, 0], dtype=np.int32)

                # Get boundary positions
                boundary_positions = {
                    'Q': [float(col) for col in boundary_Q.columns[1:]],
                    'y': [float(col) for col in boundary_y.columns[1:]],
                    'c': [float(col) for col in boundary_c.columns[1:]]
                }

                # Get boundary data
                boundary_data = {
                    'Q': {'time': t_arr_Q, 'values': _to_numpy_array(boundary_Q.iloc[:, 1:].T)},
                    'y': {'time': t_arr_y, 'values': _to_numpy_array(boundary_y.iloc[:, 1:].T)},
                    'c': {'time': t_arr_c, 'values': _to_numpy_array(boundary_c.iloc[:, 1:].T)}
                }

                # Set boundary conditions
                model.set_boundary_conditions(
                    boundary_positions, boundary_data)
            except Exception as e:
                print(f"Warning: Could not load boundary conditions: {e}")

            try:
                # Load lateral inflows
                lateral_Q = pd.read_csv(os.path.join(input_folder, 'lateral_Q.csv'),
                                        sep=';', encoding='utf-8', header=[0, 1])
                lateral_c = pd.read_csv(os.path.join(input_folder, 'lateral_c.csv'),
                                        sep=';', encoding='utf-8', header=[0, 1])

                # Process lateral inflows
                t_arr_lQ = _to_numpy_array(
                    lateral_Q.iloc[:, 0], dtype=np.int32)
                t_arr_lc = _to_numpy_array(
                    lateral_c.iloc[:, 0], dtype=np.int32)

                # Get lateral segments - handle MultiIndex columns from CSV
                lateral_Q_segments = []
                for col in lateral_Q.columns[1:]:
                    if isinstance(col, tuple):
                        # MultiIndex column - first element is the actual column name
                        col_name = col[0]
                        if isinstance(col_name, str) and col_name.startswith("('") and col_name.endswith("')"):
                            # Parse string representation of tuple like "('500', '600')"
                            # Remove outer parentheses and quotes, then split by comma
                            inner = col_name.strip("()").replace("'", "")
                            parts = [p.strip() for p in inner.split(",")]
                            lateral_Q_segments.append(
                                (int(parts[0]), int(parts[1])))
                        else:
                            # Try to parse as comma-separated values
                            parts = str(col_name).split(',')
                            lateral_Q_segments.append(
                                (int(parts[0].strip()), int(parts[1].strip())))
                    else:
                        # Single level column
                        if isinstance(col, str) and col.startswith("('") and col.endswith("')"):
                            inner = col.strip("()").replace("'", "")
                            parts = [p.strip() for p in inner.split(",")]
                            lateral_Q_segments.append(
                                (int(parts[0]), int(parts[1])))
                        else:
                            parts = str(col).split(',')
                            lateral_Q_segments.append(
                                (int(parts[0].strip()), int(parts[1].strip())))

                lateral_c_segments = []
                for col in lateral_c.columns[1:]:
                    if isinstance(col, tuple):
                        # MultiIndex column - first element is the actual column name
                        col_name = col[0]
                        if isinstance(col_name, str) and col_name.startswith("('") and col_name.endswith("')"):
                            # Parse string representation of tuple like "('500', '600')"
                            # Remove outer parentheses and quotes, then split by comma
                            inner = col_name.strip("()").replace("'", "")
                            parts = [p.strip() for p in inner.split(",")]
                            lateral_c_segments.append(
                                (int(parts[0]), int(parts[1])))
                        else:
                            # Try to parse as comma-separated values
                            parts = str(col_name).split(',')
                            lateral_c_segments.append(
                                (int(parts[0].strip()), int(parts[1].strip())))
                    else:
                        # Single level column
                        if isinstance(col, str) and col.startswith("('") and col.endswith("')"):
                            inner = col.strip("()").replace("'", "")
                            parts = [p.strip() for p in inner.split(",")]
                            lateral_c_segments.append(
                                (int(parts[0]), int(parts[1])))
                        else:
                            parts = str(col).split(',')
                            lateral_c_segments.append(
                                (int(parts[0].strip()), int(parts[1].strip())))

                # Get lateral data
                lateral_Q_data = {
                    'time': t_arr_lQ,
                    'values': _to_numpy_array(lateral_Q.iloc[:, 1:].T)
                }

                lateral_c_data = {
                    'time': t_arr_lc,
                    'values': _to_numpy_array(lateral_c.iloc[:, 1:].T)
                }

                # Set lateral inflows
                model.set_lateral_inflows(
                    lateral_Q_segments, lateral_Q_data,
                    lateral_c_segments, lateral_c_data
                )
            except Exception as e:
                print(f"Warning: Could not load lateral inflows: {e}")

            return model
        except Exception as e:
            raise ValueError(f"Error loading model from Excel: {e}") from e


# For backwards compatibility
def run_simulation():
    """
    Run the SIHQUAL simulation using the legacy implementation.

    This function is provided for backwards compatibility with the old API.
    It is recommended to use the SIHQUAL class instead.

    Returns:
        DataFrame with simulation results
    """
    print("DEPRECATED: This function is deprecated. Please use the SIHQUAL class instead.")
    print("Example:")
    print("    model = SIHQUAL(dx=100, dt=10, xf=1000, tf=3600)")
    print("    model.set_uniform_geometry(bottom_width=10, side_slope=0, manning_coef=0.03, bed_slope=0.001)")
    print("    model.set_uniform_initial_conditions(depth=2, velocity=0.5)")
    print("    results = model.run()")

    # Create a simple model
    model = SIHQUAL.create_simple_channel(length=1000, simulation_time=3600)

    # Run the simulation
    return model.run()
