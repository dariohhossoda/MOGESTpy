import pandas as pd
import numpy as np


# Ler dados -> pandas
filename = '.xlsx'

data_sheet = 'Data'
param_sheet = 'Parameters'

data_df = pd.read_excel(filename, sheet_name=data_sheet)
param_df = pd.read_excel(filename, sheet_name=param_sheet)

dx = param_df[0][1]
dt = param_df[1][1]

alpha = param_df[2][1]
coef_D = param_df[3][1]


# Guardar em variáveis
# Inicializa variáves (vetores -> numpy)
# y, v, QL, b, m, n, So, alpha, dt, dx, g
# Interopolação dos dados -> diário:horário
# Output das seções
# Loop numérico


