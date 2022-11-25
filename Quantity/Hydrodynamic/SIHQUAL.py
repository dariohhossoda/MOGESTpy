import pandas as pd
import numpy as np


filename = 'SIHQUAL.xlsx'

data_sheet = 'Data'
param_sheet = 'Parameters'

data_df = pd.read_excel(filename, sheet_name=data_sheet)
param_df = pd.read_excel(filename, sheet_name=param_sheet)

dx = param_df[0][1]
dt = param_df[1][1]

tf = param_df[2][1]

alpha = param_df[3][1]
coef_D = param_df[4][1]

x = data_df['x']
So = data_df['So']
m = data_df['m']
y1 = data_df['y1']
v1 = data_df['v1']
b = data_df['b']
B1 = data_df['B1']
A1 = data_df['A1']
Rh1 = data_df['Rh1']
Sf1 = data_df['Sf1']
Q1 = data_df['Q1']
n = data_df['n']

Kd = data_df['Kd']
Ks = data_df['Ks']
c1 = data_df['c1']
cqd = data_df['cqd']

output_sections = param_df['output_sections']

output_df = pd.DataFrame()

# Loop numérico
for sim_time in range(stop=tf, step=dt):
    # region Contribuição Lateral
    
    # endregion Contribuição Lateral
    
    # region Módulo Hidrodinâmico
    
    # endregion Módulo Hidrodinâmico
    
    # region Módulo de Qualidade
    
    # endregion Módulo de Qualidade

    # region Redefinição de Variáveis
    
    # endregion Redefinição de Variáveis
    
    # region Condição de Estabilidade
    
    # endregion Condição de Estabilidade
    
    # region Output
    
    # endregion Output
    
    pass

output_df.to_excel('SIHQUAL_output.xlsx')