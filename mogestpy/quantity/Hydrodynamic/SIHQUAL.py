"""
Simulação Hidrodinâmica e de Qualidade da Água (SIHQUAL)
em Python

DEPRECATED: This module is deprecated and will be removed in a future version.
Please use the new class-based implementation in mogestpy.quantity.Hydrodynamic.sihqual instead.
"""
import os
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd

# Show deprecation warning
warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use the new class-based implementation in mogestpy.quantity.Hydrodynamic.sihqual instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the new implementation for compatibility
try:
    from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL as SIHQUALClass
except ImportError:
    pass

cwd = os.getcwd()
filename = os.path.join(cwd, 'Quantity/Hydrodynamic/SIHQUAL.xlsx')

data_sheet = 'Data'
param_sheet = 'Parameters'

try:
    data_df = pd.read_excel(filename, sheet_name=data_sheet)
    param_df = pd.read_excel(filename, sheet_name=param_sheet)

    dx = param_df['dx'][0]
    dt = param_df['dt'][0]

    xf = param_df['xf'][0]
    tf = param_df['tf'][0]

    alpha = param_df['alpha'][0]
    DD = param_df['D'][0]

    x = data_df['x'].to_numpy(dtype=np.float64)

    b1 = data_df['b1'].to_numpy(dtype=np.float64)
    y1 = data_df['y1'].to_numpy(dtype=np.float64)
    m = data_df['m'].to_numpy(dtype=np.float64)

    n = data_df['n'].to_numpy(dtype=np.float64)
    So = data_df['So'].to_numpy(dtype=np.float64)
    v1 = data_df['v1'].to_numpy(dtype=np.float64)

    Kd = data_df['kd'].to_numpy(dtype=np.float64)
    Ks = data_df['ks'].to_numpy(dtype=np.float64)
    c1 = data_df['c1'].to_numpy(dtype=np.float64)
    cqd = data_df['cq'].to_numpy(dtype=np.float64)

    input_folder = os.path.join(cwd, 'Quantity/Hydrodynamic/SIHQUALInputs/')
    dim = int(xf / dx) + 1

    try:
        boundary_Q = pd.read_csv(input_folder+'boundary_Q.csv',
                                sep=';', encoding='utf-8')
        boundary_y = pd.read_csv(input_folder+'boundary_y.csv',
                                sep=';', encoding='utf-8')
        boundary_c = pd.read_csv(input_folder+'boundary_c.csv',
                                sep=';', encoding='utf-8')

        t_arr_Q = boundary_Q.iloc[:, 0].to_numpy(dtype=np.int32)
        t_arr_y = boundary_y.iloc[:, 0].to_numpy(dtype=np.int32)
        t_arr_c = boundary_c.iloc[:, 0].to_numpy(dtype=np.int32)
    except Exception as e:
        print(f"Error loading boundary conditions: {e}")
        t_arr_Q = None
        t_arr_y = None
        t_arr_c = None
except Exception as e:
    print(f"Error loading input files: {e}")


def ifromx(x, L, dim):
    """
    Convert position to array index.
    
    Args:
        x: Position
        L: Total length
        dim: Number of points
        
    Returns:
        Index in the array
    """
    r_index = int(x / L * (dim - 1))
    return r_index if isinstance(x, (int, float)) else [r_index]


try:
    if t_arr_Q is not None:
        index_Q = ifromx(np.array([float(col) for col in boundary_Q.columns[1:]]), xf, dim)
        index_y = ifromx(np.array([float(col) for col in boundary_y.columns[1:]]), xf, dim)
        index_c = ifromx(np.array([float(col) for col in boundary_c.columns[1:]]), xf, dim)

        bQ_data = boundary_Q.iloc[:, 1:].T.to_numpy()
        by_data = boundary_y.iloc[:, 1:].T.to_numpy()
        bc_data = boundary_c.iloc[:, 1:].T.to_numpy()
except Exception as e:
    print(f"Error processing boundary conditions: {e}")

try:
    lateral_Q = pd.read_csv(input_folder+'lateral_Q.csv',
                            sep=';', encoding='utf-8', header=[0, 1])
    t_arr_lQ = lateral_Q.iloc[:, 0].to_numpy(dtype=np.int32)
    v_tuples = lateral_Q.columns[1:][:]
    lQ_slices = [(int(int(a) / xf * (dim - 1)),
                  (int(int(b) / xf * (dim - 1)))) for a, b in v_tuples]
    lQ_data = lateral_Q.iloc[:, 1:].T.to_numpy()
except Exception as e:
    print(f"Error loading lateral flow: {e}")
    lQ_slices = []

try:
    lateral_c = pd.read_csv(input_folder+'lateral_c.csv',
                            sep=';', encoding='utf-8', header=[0, 1])
    t_arr_lc = lateral_c.iloc[:, 0].to_numpy(dtype=np.int32)
    v_tuples = lateral_c.columns[1:][:]
    lc_slices = [(int(int(a) / xf * (dim - 1)),
                  (int(int(b) / xf * (dim - 1)))) for a, b in v_tuples]
    lc_data = lateral_c.iloc[:, 1:].T.to_numpy()
except Exception as e:
    print(f"Error loading lateral concentration: {e}")
    lc_slices = []

try:
    sections = param_df['output_sections'].to_numpy()
except Exception as e:
    print(f"Error loading output sections: {e}")
    sections = []

g = 9.81

# region Aux


def avg(vector):
    """
    Vetor médio.

    Corresponde a média centrrada (i + 1 e i - 1).
    """
    return .5 * (vector[2:] + vector[:-2])


def cdelta(vector):
    """
    Diferença centrada.

    Corresponde a operação vector[i+1] - vector[i-1]
    para i variando de 1 a len(vector) - 1.
    """
    return vector[2:] - vector[:-2]


def top_base(b, y, m):
    """
    Largura do topo - vetorial
    """
    return b + 2 * m * y


def wet_area(b, y, m):
    """
    Área molhada
    """
    return b * y + m * y * y


def wet_perimeter(b, y, m):
    """
    Perímetro molhado
    """
    return b + 2 * y * np.sqrt(1 + m ** 2)


def Rh(b, y, m):
    """
    Raio Hidráulico
    """
    return wet_area(b, y, m) / wet_perimeter(b, y, m)


def Sf(n, v, Rh):
    """
    Declividade da linha de energia
    """
    return n * n * v * v * Rh ** (- 4 / 3)


def courant(dt, dx, v, g, A, B):
    """
    Estabilidade de Courant
    """
    return dt / dx * (np.abs(v) + (g * A / B) ** .5)
# endregion Aux


def run_simulation():
    """
    Run the SIHQUAL simulation using the legacy implementation.
    
    Returns:
        DataFrame with simulation results
    """
    auto_step = False
    courant_max = 0.0

    y2 = np.zeros_like(y1)
    v2 = np.zeros_like(v1)
    c2 = np.zeros_like(c1)

    ql = np.zeros(dim)
    cqd = np.zeros(dim)

    progress = tqdm(total=tf,
                    desc='SIHQUAL',
                    unit='s_(sim)')

    n_index = 0
    sim_time = 0

    days_total = tf // 86400 + 1
    variables = ['Q', 'y', 'c']
    cols = pd.MultiIndex.from_product(
        [sections, variables], names=['section', 'variables'])
    t_index = pd.Index(list(range(days_total)), name='t')

    col_num = len(variables) * len(sections)
    df_data = np.zeros((days_total, col_num))

    while sim_time <= tf:
        A1 = b1 * y1 + m * y1 * y1  # wet_area(b1, y1, m)
        B1 = b1 + 2 * m * y1  # top_base(b1, y1, m)
        Rh1 = A1 / (b1 + 2 * y1 * np.sqrt(1 + m ** 2))  # Rh(b1, y1, m)
        Sf1 = n * n * v1 * v1 * Rh1 ** (- 4 / 3)  # Sf(n, v1, Rh1)

        if auto_step and courant_max > 0:
            dt = .5 * dt / courant_max

        # region Contribuição Lateral
        for i, (start, end) in enumerate(lQ_slices):
            ql[slice(start, end)] = np.interp(
                sim_time + dt, t_arr_lQ, lQ_data[i])

        for i, (start, end) in enumerate(lc_slices):
            cqd[slice(start, end)] = np.interp(
                sim_time + dt, t_arr_lc, lc_data[i])
        
        # Avoid division by zero
        mask = A1 > 0
        cq = np.zeros_like(cqd)
        cq[mask] = cqd[mask] / A1[mask]
        # endregion Contribuição Lateral

        # region ModuloHidrodinamico
        yy, dy = avg(y1), cdelta(y1)
        vv, dv = avg(v1), cdelta(v1)
        AA, dA = avg(A1), cdelta(A1)
        BB = avg(B1)
        SSf = avg(Sf1)

        mfactor = - .5 * dt / dx
        y2[1:-1] = (alpha * y1[1:-1] + (1 - alpha) * yy
                    + mfactor * vv * dy
                    + mfactor * vv * dA / BB
                    + mfactor * AA * dv / BB
                    + ql[1:-1] * dt / BB)

        v2[1:-1] = (alpha * v1[1:-1] + (1 - alpha) * vv
                    + mfactor * vv * dv
                    + mfactor * g * dy
                    + g * dt * (So[1:-1] - SSf))

        y2[-1] = y2[-2]
        v2[-1] = v2[-2]
        # endregion ModuloHidrodinamico

        # region Módulo de Qualidade
        dc = cdelta(c1)

        c2[1:-1] = (c1[1:-1]
                    - .5 * dt / dx * v1[1:-1] * dc
                    + DD * .5 * dt / dx / A1[1:-1] * dA * dc * .5 / dx
                    + DD * dt / dx ** 2
                    )

        c2[-1] = c2[-2]
        # endregion Módulo de Qualidade

        # region Redefinição de Variáveis
        for i, idx in enumerate(index_y):
            y2[idx] = np.interp(sim_time + dt, t_arr_y, by_data[i])
        
        A2 = b1 * y2 + m * y2 * y2
        
        for i, idx in enumerate(index_Q):
            if A2[idx] > 0:
                v2[idx] = np.interp(sim_time + dt, t_arr_Q, bQ_data[i]) / A2[idx]
        
        for i, idx in enumerate(index_c):
            c2[idx] = np.interp(sim_time + dt, t_arr_c, bc_data[i])

        y1 = np.copy(y2)
        v1 = np.copy(v2)
        c1 = np.copy(c2)
        # endregion Redefinição de Variáveis

        courant_max = (dt / dx * (np.abs(v1) + np.sqrt(g * A1 / B1))).max()
        if courant_max >= 1:
            raise ValueError(f'Stability error! Courant number = {courant_max:.4f} >= 1.0')

        # region Output
        days_elapsed = sim_time / 86400
        if days_elapsed >= n_index:
            k = 0
            _array = np.zeros(col_num)
            for section in sections:
                _i = int(int(section) / xf * (dim - 1))
                _array[k] = v1[_i] * A1[_i]
                _array[k+1] = y1[_i]
                _array[k+2] = c1[_i]
                k += 3
            df_data[int(days_elapsed)] = _array

            n_index += 1
        # endregion Output

        sim_time += dt
        progress.update(n=dt)

    progress.close()
    output_df = pd.DataFrame(df_data, columns=cols, index=t_index)
    return output_df


def main():
    """
    Main function to run the simulation and save results.
    """
    print("DEPRECATED: This module is deprecated. Please use the new class-based implementation.")
    print("For example:")
    print("    from mogestpy.quantity.Hydrodynamic.sihqual import SIHQUAL")
    print("    model = SIHQUAL.from_excel('SIHQUAL.xlsx')")
    print("    results = model.run()")
    print("    model.save_results(results, 'output.xlsx')")
    
    try:
        output_df = run_simulation()
        try:
            output_df.to_excel('SIHQUAL_output.xlsx')
            print("Results saved to 'SIHQUAL_output.xlsx'")
        except Exception as e:
            print(f'Error saving results: {e}')
    except Exception as e:
        print(f"Error running simulation: {e}")


if __name__ == "__main__":
    main()