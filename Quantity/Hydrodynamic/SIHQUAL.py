"""
Simulação Hidrodinâmica e de Qualidade da Água (SIHQUAL)
em Python
"""
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
# from numpy.core.multiarray import interp as compiled_interp

cwd = os.getcwd()
filename = os.path.join(cwd, 'Quantity/Hydrodynamic/SIHQUAL.xlsx')

data_sheet = 'Data'
param_sheet = 'Parameters'

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

input_folder = os.path.join(cwd,'Quantity/Hydrodynamic/SIHQUALInputs/')
dim = int(xf / dx) + 1

boundary_Q = pd.read_csv(input_folder+'boundary_Q.csv', sep=';', encoding='utf-8')
boundary_y = pd.read_csv(input_folder+'boundary_y.csv', sep=';', encoding='utf-8')
boundary_c = pd.read_csv(input_folder+'boundary_c.csv', sep=';', encoding='utf-8')

t_arr_Q = boundary_Q.iloc[:, 0].to_numpy(dtype=np.int32)
t_arr_y = boundary_y.iloc[:, 0].to_numpy(dtype=np.int32)
t_arr_c = boundary_c.iloc[:, 0].to_numpy(dtype=np.int32)

def ifromx (x, L, dim):
    r_index = int(x / L * (dim - 1))
    return  r_index if x.size > 1 else [r_index] 

index_Q = ifromx(boundary_Q.columns[1:].to_numpy(dtype=np.int32), xf, dim)
index_y = ifromx(boundary_y.columns[1:].to_numpy(dtype=np.int32), xf, dim)
index_c = ifromx(boundary_c.columns[1:].to_numpy(dtype=np.int32), xf, dim)

bQ_data = boundary_Q.iloc[:, 1:].T.to_numpy()
by_data = boundary_y.iloc[:, 1:].T.to_numpy()
bc_data = boundary_c.iloc[:, 1:].T.to_numpy()

try:
    lateral_Q = pd.read_csv(input_folder+'lateral_Q.csv', sep=';', encoding='utf-8', header=[0, 1])
    t_arr_lQ = lateral_Q.iloc[:, 0].to_numpy(dtype=np.int32)
    v_tuples = lateral_Q.columns[1:][:]
    lQ_slices = [(int(int(a)/ xf * (dim - 1)),
                  (int(int(b)/ xf * (dim - 1)))) for a, b in v_tuples]
    lQ_data = lateral_Q.iloc[:, 1:].T.to_numpy()
except:
    pass
try:
    lateral_c = pd.read_csv(input_folder+'lateral_c.csv', sep=';', encoding='utf-8', header=[0, 1])
    t_arr_lc = lateral_c.iloc[:, 0].to_numpy(dtype=np.int32)
    v_tuples = lateral_c.columns[1:][:]
    lc_slices = [(int(int(a)/ xf * (dim - 1)),
                  (int(int(b)/ xf * (dim - 1)))) for a, b in v_tuples]
    lc_data = lateral_c.iloc[:, 1:].T.to_numpy()
except:
    pass


g = 9.81

t_out = []
y_out = []
Q_out = []

output_sections = param_df['output_sections']

output_df = pd.DataFrame()

# region Aux
def v_manning(n, Rh, So):
    return 1 / n * Rh ** (2/3) * So ** .5

def Q_manning(n, Rh, So, A):
    return v_manning(n, Rh, So) * A

def df2array(column_name,
             dataframe,
             index_array,
             index_name = 'x'):
    aux = dataframe[[index_name, column_name]].dropna()
    return np.interp(index_array, aux[index_name], aux[column_name])

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
    return b + 2 * m  * y

def wet_area(b, y, m):
    """
    Área molhada
    """
    return b * y + m * y * y

def wet_perimeter(b, y, m):
    """
    Perímetro molhado
    """
    return b + 2 * y * (1 + m ** 2) ** .5

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
    return dt / dx * (abs(v) + (g * A / B) ** .5)
# endregion Aux

# Inicialização dos vetores de contribuição lateral

auto_step = False

y2 = np.zeros_like(y1)
v2 = np.zeros_like(v1)
c2 = np.zeros_like(c1)

ql = np.zeros(dim)
cqd = np.zeros(dim)
#%%
progress = tqdm(total=tf,
                desc='SIHQUAL',
                unit='s_(sim)')
n_index = 0
sim_time = 0
while sim_time <= tf:
    A1 = b1 * y1 + m * y1 * y1 # wet_area(b1, y1, m)
    B1 = b1 + 2 * m  * y1 # top_base(b1, y1, m)
    Rh1 = A1 / (b1 + 2 * y1 * (1 + m ** 2) ** .5) # Rh(b1, y1, m)
    Sf1 =  n * n * v1 * v1 * Rh1 ** (- 4 / 3) # Sf(n, v1, Rh1)

    if auto_step:
        dt = .5 * dt / courant_max

    # region Contribuição Lateral
    for i in range(len(lQ_slices)):
        ql[slice(lQ_slices[i][0], lQ_slices[i][1])] = np.interp(sim_time + dt, t_arr_lQ, lQ_data[i])
    
    for i in range(len(lc_slices)):
        cqd[slice(lc_slices[i][0], lc_slices[i][1])] = np.interp(sim_time + dt, t_arr_lQ, lc_data[i])
    cq = cqd / A1
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
                + DD * .5 *  dt / dx  / A1[1:-1] * dA * dc * .5 / dx
                + DD * dt / dx ** 2
                )

    c2[-1] = c2[-2]
    # endregion Módulo de Qualidade

    # region Redefinição de Variáveis
    y2[index_y] = [np.interp(sim_time + dt, t_arr_y, by_data[i]) for i in index_y]
    A2 = b1 * y2 + m * y2 * y2
    v2[index_Q] = [np.interp(sim_time + dt, t_arr_y, bQ_data[i]) / A2[i] for i in index_Q]
    c2[index_c] = [np.interp(sim_time + dt, t_arr_y, bc_data[i]) for i in index_c]
    
    y1 = np.copy(y2)
    v1 = np.copy(v2)
    c1 = np.copy(c2)
    # endregion Redefinição de Variáveis

    courant_max = dt / dx * (abs(v1) + (g * A1 / B1) ** .5).max()
    if courant_max >= 1:
        raise Exception('Erro de estabilidade!')

    # region Output
    days_elapsed = sim_time / 86400
    if  days_elapsed >= n_index:
        t_out.append(int(days_elapsed))
        y_out.append(y1[0])
        Q_out.append(v1[0] * A1[0])

        n_index += 1
    # endregion Output

    sim_time +=  dt
    progress.update(n = dt)

output_df['t'] = t_out
output_df['y'] = y_out
output_df['Q'] = Q_out

output_df.to_excel('SIHQUAL_output.xlsx', index=False)
