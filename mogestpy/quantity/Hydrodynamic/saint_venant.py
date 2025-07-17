"""
Saint-Venant equations for one-dimensional hydrodynamic routing
with trapezoidal cross-section.

This module implements the Saint-Venant equations for one-dimensional
hydrodynamic flow routing using a trapezoidal cross-section.
"""
import math


class TrapezoidalCrossSection:
    """
    Trapezoidal cross-section for channel flow calculations.

    This class represents a trapezoidal cross-section with methods to calculate
    geometric properties such as wet area, wet perimeter, and hydraulic radius.

    Attributes:
        b (float): Bottom width of the channel (m)
        y (float): Water depth (m)
        m (float): Side slope (horizontal/vertical)
        top_width (float): Width at the water surface (m)
        hydraulic_radius (float): Hydraulic radius (m)
    """

    def __init__(self, b: float, y: float, m: float):
        """
        Initialize a trapezoidal cross-section.

        Args:
            b (float): Bottom width of the channel (m)
            y (float): Water depth (m)
            m (float): Side slope (horizontal/vertical)
        """
        self.b = b
        self.y = y
        self.m = m

        self.top_width = self.calculate_top_width()
        self.hydraulic_radius = self.wet_area() / self.wet_perimeter()

    def __str__(self):
        """
        String representation of the trapezoidal cross-section.

        Returns:
            str: Formatted string with cross-section parameters
        """
        return f'Trapezoidal Cross Section:\n\
b: {self.b:.3f}\n\
y: {self.y:.3f}\n\
m: {self.m:.3f}'

    def calculate_top_width(self) -> float:
        """
        Calculate the width at the water surface.

        Returns:
            float: Top width of the water surface (m)
        """
        return self.b + 2 * self.m * self.y

    def wet_area(self) -> float:
        """
        Calculate the wet area of the cross-section.

        Returns:
            float: Wet area (m²)
        """
        return self.b * self.y + self.m * self.y ** 2

    def wet_perimeter(self) -> float:
        """
        Calculate the wet perimeter of the cross-section.

        Returns:
            float: Wet perimeter (m)
        """
        return self.b + 2 * self.y * math.sqrt(1 + self.m ** 2)

    def normal_depth(self, discharge, manning_n, slope) -> float:
        """
        Calculate the normal depth using the Newton-Raphson method.

        Args:
            discharge (float): Flow discharge (m³/s)
            manning_n (float): Manning's roughness coefficient
            slope (float): Channel bed slope (m/m)

        Returns:
            float: Normal depth (m)
        """
        y_norm = 0.1
        f = 1
        exp = 2/3
        tolerance = 1e-13
        steps = 0
        max_steps = 100

        area = self.wet_area()
        perimeter = self.wet_perimeter()

        while math.fabs(f) > tolerance and steps < max_steps:
            f = (area ** (1 + exp) / perimeter ** exp
                 - discharge * manning_n / slope ** 0.5)
            df = (5 * area ** exp *
                  (self.b + 2 * self.m * y_norm) / perimeter ** exp
                  - 4 * area ** (1 + exp) / perimeter ** (1 + exp)
                  * (1 + self.m ** 2) ** 0.5 / 3)
            y_norm -= f / df
            steps += 1

        return y_norm

    def area_depth(self, area):
        """
        Calculate the water depth for a given wet area using Newton-Raphson method.

        Args:
            area (float): Wet area (m²)

        Returns:
            float: Water depth (m)
        """
        y = 1
        step = 0
        f = 1
        tolerance = 1e-13
        max_steps = 100

        while math.fabs(f) > tolerance and step < max_steps:
            f = self.b * y + self.m * y ** 2
            df = self.b + 2 * self.m * y
            y -= (f - area) / df
            step += 1

        return y


class SaintVenant:
    """
    Saint-Venant equations for one-dimensional hydrodynamic routing.

    This class implements the Saint-Venant equations for one-dimensional
    hydrodynamic flow routing with a trapezoidal cross-section.

    Attributes:
        cross_section (TrapezoidalCrossSection): Channel cross-section
        discharge (float): Flow discharge (m³/s)
        manning_n (float): Manning's roughness coefficient
        slope (float): Channel bed slope (m/m)
        dt (float): Time step (s)
        dx (float): Space step (m)
        g (float): Gravitational acceleration (m/s²)
    """

    def __init__(self, cross_section, discharge, manning_n, slope, dt, dx, g=9.81):
        """
        Initialize the Saint-Venant model.

        Args:
            cross_section (TrapezoidalCrossSection): Channel cross-section
            discharge (float): Flow discharge (m³/s)
            manning_n (float): Manning's roughness coefficient
            slope (float): Channel bed slope (m/m)
            dt (float): Time step (s)
            dx (float): Space step (m)
            g (float, optional): Gravitational acceleration (m/s²). Defaults to 9.81.
        """
        self.cross_section = cross_section
        self.discharge = discharge
        self.manning_n = manning_n
        self.slope = slope
        self.dt = dt
        self.dx = dx
        self.g = g

    def courant(self):
        """
        Calculate the Courant number.

        Returns:
            float: Courant number (dimensionless)
        """
        area = self.cross_section.wet_area()
        return (self.dt / self.dx *
                (self.g * self.cross_section.y + self.discharge / area) ** 0.5)

    def courant_check(self) -> bool:
        """
        Check if the Courant condition for numerical stability is satisfied.

        Returns:
            bool: True if Courant number <= 1, False otherwise
        """
        return self.courant() <= 1

    def quadratic_froude(self) -> float:
        """
        Calculate the square of the Froude number.

        Returns:
            float: Square of the Froude number (dimensionless)
        """
        area = self.cross_section.wet_area()
        return (self.discharge ** 2 * self.cross_section.top_width /
                (self.g * area ** 3))

    def friction_slope(self) -> float:
        """
        Calculate the friction slope (energy grade line slope).

        Returns:
            float: Friction slope (m/m)
        """
        area = self.cross_section.wet_area()
        hydraulic_radius = self.cross_section.hydraulic_radius

        return (self.discharge * math.fabs(self.discharge) * self.manning_n ** 2 /
                (area ** 2 * hydraulic_radius ** (4 / 3)))

    def update_values(self):
        """
        Update the state based on the time step.

        Raises:
            NotImplementedError: Method not yet implemented
        """
        # for i in range(1, self.y ):
        raise NotImplementedError('Not yet implemented!')

    def run_model(self):
        """
        Run the Saint-Venant model.

        Raises:
            NotImplementedError: Method not yet implemented
        """
        raise NotImplementedError('Not yet implemented!')


def lateral_contribution():
    """
    Define the lateral contribution to be included in the model.

    Raises:
        NotImplementedError: Function not yet implemented
    """
    raise NotImplementedError('Not yet implemented!')


def average(values_list, index):
    """
    Calculate the centered average of values at a given index.

    Args:
        values_list (list): List of values
        index (int): Index for which to calculate the centered average

    Returns:
        float: Centered average of values at index-1 and index+1,
               or None if index is out of bounds
    """
    try:
        return 0.5 * (values_list[index + 1] + values_list[index - 1])
    except IndexError:
        return None
