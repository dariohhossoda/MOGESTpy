import os

import pandas as pd
import numpy as np

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
coef_D = param_df['D'][0]

x = data_df['x']
So = data_df['So']
m = data_df['m']
y1 = data_df['y1']
v1 = data_df['v1']
b = data_df['b1']
B1 = data_df['B1']
A1 = data_df['A1']
Rh1 = data_df['Rh1']
Sf1 = data_df['Sf1']
Q1 = data_df['Q1']
n = data_df['n']

Kd = data_df['kd']
Ks = data_df['ks']
c1 = data_df['c1']
cqd = data_df['cq']

output_sections = param_df['output_sections']

output_df = pd.DataFrame()

def vec_avg(vector):
    return (vector[i - 1] + vector [i + 1]) * .5
                                
                        
sim_time = 0
auto_step = True
g = 9.81
# Loop numérico
while sim_time <= tf:
    
    
    # TODO: Implementar contribuicao lateral
    # region Contribuição Lateral
    cq = cqd / A1
    # endregion Contribuição Lateral
    
    # region ModuloHidrodinamico
    
    y2 = y1.copy()
    v2 = y2.copy()
    for i in range(1, len(y1) - 1):
        yy = vec_avg(y1)
        vv = vec_avg(v1)
        SSf = vec_avg(Sf1)
        AA = vec_avg(A1)
        BB = vec_avg(B1)
        # region Continuity Eq
        
        y2[i] = (alpha * y1[i] + (1 - alpha) * yy
                    - (0.5 * dt / dx / BB) * (vv * (y1[i + 1] - y1[i - 1]) * BB
                    + vv * (A1[i + 1] - A1[i - 1])
                    + AA * (v1[i + 1] - v1[i - 1]))
                    + ql[i] * dt / BB)
        # endregion Continuity Eq
        # region Momentum Eq
        v2[i] = (alpha * v1[i] + (1 - alpha) * vv
                    - 0.5 * dt / dx * ( vv * (v1[i + 1] - v1[i - 1])
                    + g * (y1[i + 1] - y1[i - 1]) )
                    + g * dt * (So[i] - SSf))
        # endregion Momentum Eq
        
        y2[-1] = y2[-2]
        v2[-1] = v2[-2]
        
        
    # endregion ModuloHidrodinamico
    
    # region Módulo de Qualidade
    
    # endregion Módulo de Qualidade

    # region Redefinição de Variáveis
    
    # endregion Redefinição de Variáveis
    
    # region Condição de Estabilidade
    
    # endregion Condição de Estabilidade
    
    # region Output
    
    # endregion Output
    
    sim_time += dt if not auto_step else dt

output_df.to_excel('SIHQUAL_output.xlsx')