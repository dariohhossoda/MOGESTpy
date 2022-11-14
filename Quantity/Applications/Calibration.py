import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from Otimizacoes import *
from Upstream import *
import Quantity.Hydrological.Muskingum as Muskingum
import Quantity.Hydrological.SMAP as SMAP


def Calibracao(obsAtibaia, obsValinhos, revAtibainha,
               revCachoeira, Atibaia, Valinhos, FO):
    """
    1. Muskingum de jusante até o ponto de controle de Atibaia:
    O modelo utilizará como parâmetros, duas trincas de variáveis Muskingum
    (K1, X1, m1) p/ Atibainha e (K2, X2, m2) p/ Cachoeira e também uma quina
    de variáveis SMAP (Str, k2t, Crec, TUin, EBin). O objetivo será minimizar
    as diferenças entre as vazões incrementais calculadas com o routing hidrológico
    e aquelas obtidas com o módulo chuva-vazão
    Condições de contorno para variáveis que serão calibradas

    Parâmetros:
    ----
    obsAtibaia, obsValinhos : Captações, chuvas e vazões observadas nos pontos
    revAtibainha, revCachoeira : Despachos observados nos reservatórios
    Atibaia, Valinhos : Bacias
    FO : Função objetivo p/ otimizações
    """

    bounds = [[1000.0, 2000.0],  # Str
              [0.2, 4.0],  # k2t
              [0.0, 1.0],  # Crec
              [0.0, 1.0],  # TUin
              [0.0, 9.2],  # EBin
              [60, 80],  # K1 (entre 2.5 e 3.3 dias)
              [0.2, 0.5],  # X1
              [1.1, 1.3],  # m1 (forçando o modelo a não escolher m = 1)
              [60, 80],  # K2 (entre 2.5 e 3.3 dias)
              [0.2, 0.5],  # X2
              [1.1, 1.3]]  # m2 (forçando o modelo a não escolher m = 1)

    n = len(obsAtibaia.Q)

    # Função objetivo
    def objective(p):
        # Sujeitos a calibração
        Str, k2t, Crec, TUin, EBin, K1, X1, m1, K2, X2, m2 = p

        # Routing de jusante não linear 1: de Atibainha para Atibaia
        Q1 = Muskingum.DownstreamFORK(K1, X1, m1, 24.0, revAtibainha.D)
        # Routing de jusante não linear 2: de Cachoeira para Atibaia
        Q2 = Muskingum.DownstreamFORK(K2, X2, m2, 24.0, revCachoeira.D)

        # Junto ao ponto de controle, a vazão observada equivale a uma parcela
        # despachada de cada reservatório mais uma parcela incremental de eventos chuvosos
        # menos uma parcela captada entre as barragens e a própria seção
        inc1 = [0] * n
        for j in range(n):
            inc1[j] = obsAtibaia.Q[j] - (Q1[j] + Q2[j]) + obsAtibaia.C[j]

        # Segundo vetor incremental ("calc")
        inc2 = SMAP(Str, k2t, Crec, TUin, EBin, obsAtibaia, Atibaia)

        # Restrição positiva aos routings calculados e às vazões incrementais
        minQ1, minQ2 = min(Q1), min(Q2)
        res1, res2 = min(inc1), min(inc2)
        if minQ1 < 0 or minQ2 < 0 or res1 < 0 or res2 < 0:
            return np.inf
        else:
            # Métrica utilizada para otimização
            if FO == 1:
                # NSE: Nash-Sutcliffe
                return NSE(inc1, inc2)
            elif FO == 2:
                # SSQ: Sum of Squares of Deviations
                return SSQ(inc1, inc2)
            else:
                # RMSE: Root-Mean-Square Error
                return RMSE(inc1, inc2)

    # Busca por evolução diferencial
    result = differential_evolution(objective, bounds, maxiter=1000)
    # Resultados
    print('Muskingum de jusante e SMAP')
    print('Atibaia:')
    print('Status: %s' % result['message'])
    print('Avaliações realizadas: %d' % result['nfev'])
    # Solução
    solution = result['x']
    evaluation = objective(solution)
    print('Solução: \n'
          'f = ( \n'
          '\t[Str = %.3f \n\t k2t = %.3f \n\t Crec = %.3f \n\t TUin = %.3f \n\t EBin = %.3f \n\t K1 = %.3f \n\t X1 = %.3f \n\t m1 = %.3f \n\t K2 = %.3f \n\t X2 = %.3f \n\t m2 = %.3f]'
          % (
              solution[0], solution[1], solution[2], solution[3], solution[4], solution[5], solution[6], solution[7],
              solution[8], solution[9], solution[10]))
    if FO == 1:
        print(') = %.3f' % (1 - evaluation))
    else:
        print(') = %.3f' % evaluation)

    # Armazenamento em dicionário para utilização durante etapa de previsão
    # (K, X e m com final 1 referem-se a Atibainha; aqueles com final 2 são de Cachoeira)
    paramsAtibaia = {
        'Str': solution[0],
        'k2t': solution[1],
        'Crec': solution[2],
        'K1': [solution[5]],
        'X1': [solution[6]],
        'm1': [solution[7]],
        'K2': [solution[8]],
        'X2': [solution[9]],
        'm2': [solution[10]]
    }

    # 2. Checagem de incrementais e conversão chuva-vazão para o período observado em Atibaia:
    # As incrementais são necessárias para averiguar como as vazões obtidas com os parâmetros calibrados
    # adequam-se aos dados "observados" (também advindos de uma calibração própria, devido à parcela de despacho)
    newQ1 = Muskingum.DownstreamFORK(
        solution[5], solution[6], solution[7], 24.0, revAtibainha.D)
    newQ2 = Muskingum.DownstreamFORK(
        solution[8], solution[9], solution[10], 24.0, revCachoeira.D)

    incAtibaia = [0] * n
    for j in range(n):
        incAtibaia[j] = obsAtibaia.Q[j] - \
            (newQ1[j] + newQ2[j]) + obsAtibaia.C[j]

    calcAtibaia = SMAP(solution[0], solution[1], solution[2],
                       solution[3], solution[4], obsAtibaia, Atibaia)

    # 3. Muskingum de jusante até o ponto de controle de Valinhos:
    # Semelhante ao passo 1., porém com uma trinca de variáveis Muskingum (K, X, m) ao invés de duas
    # Condições de contorno para variáveis que serão calibradas
    _Str = [1000, 2000]
    _k2t = [.2, 6]
    _Crec = [0, 20]
    _TUin = [0, 1]
    _EBin = [0, 40]
    _K = [60, 120]  # Entre 2.5 e 5 dias
    _X = [.2, .5]
    _m = [1.1, 1.3]  # forçando o modelo a não escolher m = 1

    bounds = [_Str, _k2t, _Crec, _TUin, _EBin,
              _K, _X, _m]

    # Função objetivo
    def objective(p):
        # Sujeitos a calibração
        Str, k2t, Crec, TUin, EBin, K, X, m = p

        # Routing de jusante não linear: de Atibaia para Valinhos
        Q = Muskingum.DownstreamFORK(K, X, m, 24.0, obsAtibaia.Q)

        # Junto ao ponto de controle, a vazão observada equivale a uma parcela
        # despachada de cada reservatório mais uma parcela incremental de eventos chuvosos
        # menos uma parcela captada entre as barragens e a própria seção
        inc1 = [0] * n
        for j in range(n):
            inc1[j] = obsValinhos.Q[j] - Q[j] + obsValinhos.C[j]

        # Segundo vetor incremental ("calc")
        inc2 = SMAP(Str, k2t, Crec, TUin, EBin, obsValinhos, Valinhos)

        # Restrição positiva aos routings calculados e às vazões incrementais
        minQ, res1, res2 = min(Q), min(inc1), min(inc2)
        if minQ < 0 or res1 < 0 or res2 < 0:
            return np.inf
        else:
            # Métrica utilizada para otimização
            if FO == 1:
                # NSE: Nash-Sutcliffe
                return NSE(inc1, inc2)
            elif FO == 2:
                # SSQ: Sum of Squares of Deviations
                return SSQ(inc1, inc2)
            else:
                # RMSE: Root-Mean-Square Error
                return RMSE(inc1, inc2)

    # Busca por evolução diferencial
    result = differential_evolution(objective, bounds, maxiter=1000)
    # Resultados
    print()
    print('Valinhos:')
    print('Status: %s' % result['message'])
    print('Avaliações realizadas: %d' % result['nfev'])
    # Solução
    solution = result['x']
    evaluation = objective(solution)
    print('Solução: \n'
          'f = ( \n'
          '\t[Str = %.3f \n\t k2t = %.3f \n\t Crec = %.3f \n\t TUin = %.3f \n\t EBin = %.3f \n\t K = %.3f \n\t X = %.3f \n\t m = %.3f]'
          % (
              solution[0], solution[1], solution[2], solution[3], solution[4], solution[5], solution[6], solution[7]))
    if FO == 1:
        print(') = %.3f\n' % (1 - evaluation))
    else:
        print(') = %.3f\n' % evaluation)

    # Armazenamento em dicionário para utilização durante etapa de previsão
    paramsValinhos = {
        'Str': solution[0],
        'k2t': solution[1],
        'Crec': solution[2],
        'K': [solution[5]],
        'X': [solution[6]],
        'm': [solution[7]]
    }

    # 4. Checagem de incrementais e conversão chuva-vazão para o período observado em Valinhos:
    # As incrementais são necessárias para averiguar como as vazões obtidas com os parâmetros calibrados
    # adequam-se aos dados "observados" (também advindos de uma calibração própria, devido à parcela de despacho)
    newQ = Muskingum.DownstreamFORK(
        solution[5], solution[6], solution[7], 24.0, obsAtibaia.Q)

    incValinhos = [0] * n
    for j in range(n):
        incValinhos[j] = obsValinhos.Q[j] - newQ[j] + obsValinhos.C[j]

    calcValinhos = SMAP(solution[0], solution[1], solution[2],
                        solution[3], solution[4], obsValinhos, Valinhos)

    # 5. Muskingum de montante até o ponto de controle de Atibaia:
    # Depois de otimizar o módulo chuva-vazão para cada sub-bacia, é necessário tomar as
    # vazões incrementais aferidas durante o período de observação para calibrar o módulo de routing
    # inverso, ou upstream routing (de jusante para montante). Em Valinhos:
    #   Qobs.Vali. = Qinc. - capt. + Desp.Atib.
    # Portanto, convém retroceder Desp.Atib. e tomar como referência Qobs.Atib. para calibração
    desp = [0] * n
    for j in range(n):
        desp[j] = obsValinhos.Q[j] + obsValinhos.C[j] - incValinhos[j]

    # Condições de contorno para variáveis que serão calibradas
    bounds = [
        [84.0, 99.0],           # K (entre 3.5 e 4.1 dias)
        # X (necessário controlar limite inferior de X para que o modelo nao execute potenciação complexa)
        [0.2,  0.5],
        [1.1,  1.2]            # m (forçando o modelo a não escolher m = 1)
    ]  # (valores elevados de m tornam o hidrograma transladado uma linha reta, ou 'flat'. Necessário conter limite superior)

    # Função objetivo
    def objective(p):
        # Sujeitos a calibração
        K, X, m = p

        # Routing de montante não linear: de Valinhos para Atibaia
        Q = Muskingum.UpstreamFORK(K, X, m, 24.0, desp)

        # Restrição positiva ao routing calculado
        res = min(Q)
        if res < 0:
            return np.inf
        else:
            # Métrica utilizada para otimização
            if FO == 1:
                # NSE: Nash-Sutcliffe
                return NSE(obsAtibaia.Q, Q)
            elif FO == 2:
                # SSQ: Sum of Squares of Deviations
                return SSQ(obsAtibaia.Q, Q)
            else:
                # RMSE: Root-Mean-Square Error
                return RMSE(obsAtibaia.Q, Q)

    # Busca por evolução diferencial
    result = differential_evolution(objective, bounds, maxiter=1000)
    # Resultados
    print('Muskingum de montante')
    print('Valinhos:')
    print('Status: %s' % result['message'])
    print('Avaliações realizadas: %d' % result['nfev'])
    # Solução
    solution = result['x']
    evaluation = objective(solution)
    print('Solução: \n'
          'f = ( \n'
          '\t[K = %.3f \n\t X = %.3f \n\t m = %.3f]'
          % (solution[0], solution[1], solution[2]))
    if FO == 1:
        print(') = %.3f\n' % (1 - evaluation))
    else:
        print(') = %.3f\n' % evaluation)

    # Armazenamento para plotagem
    upVA = Muskingum.UpstreamFORK(
        solution[0], solution[1], solution[2], 24.0, desp)

    # Armazenamento em dicionário para utilização durante etapa de previsão
    paramsValinhos['K'] += [solution[0]]
    paramsValinhos['X'] += [solution[1]]
    paramsValinhos['m'] += [solution[2]]

    # 6. Muskingum de montante até reservatórios:
    # Depois de otimizar o módulo chuva-vazão para cada sub-bacia, é necessário tomar as
    # vazões incrementais aferidas durante o período de observação para calibrar o módulo de routing
    # inverso, ou upstream routing (de jusante para montante). Em Atibaia:
    #   Qobs.Atib. = Qinc. - capt. + (Desp.Atibain. + Desp.Cach.)
    #   (Desp.Atibain. + Desp.Cach.) = Reserv.
    # Como a descarga observada junto à seção corresponde à soma de duas parcelas, cada qual de uma barragem,
    # é preciso retroceder um percentual de Res. para comparar com Desp.Atibain.obs. O mesmo vale para
    # Res. e Desp.Cach.obs. Uma forma é realizar o routing de Res. hora multiplicado por um coeficiente
    # alfa (p/ Atibainha), hora multiplicado por beta (p/ Cachoeira). A título de simplificação, tomou-se
    # alfa = beta = 0.5
    reserv = [0] * n
    for j in range(n):
        reserv[j] = obsAtibaia.Q[j] + obsAtibaia.C[j] - incAtibaia[j]

    # Ajuste necessário para evitar exponenciação complexa por ordenada final de hidrograma próxima de 0
    minAtibainha = revAtibainha.D[n - 1]
    minCachoeira = revCachoeira.D[n - 1]
    minReserv = min(minAtibainha, minCachoeira)
    if reserv[n - 1] < minReserv:
        reserv[n - 1] = minReserv

    # Condições de contorno para variáveis que serão calibradas
    bounds = [
        [120.0, 180.0],         # K (entre 5 e 7.5 dias)
        # X (necessário controlar limite inferior para que o modelo não execute potenciação complexa)
        [0.4,   0.5],
        [1.1,   1.2]          # m (forçando o modelo a não escolher m = 1)
    ]

    # Função objetivo
    def objective(p):
        # Sujeitos a calibração
        K, X, m = p

        # Routing de montante não linear: de Atibaia para Atibainha
        Q = UpstreamFORK(K, X, m, 24.0, list(np.multiply(reserv, 0.5)))

        # Restrição positiva ao routing calculado
        res = min(Q)
        if res < 0:
            return np.inf
        else:
            # Métrica utilizada para otimização
            if FO == 1:
                # NSE: Nash-Sutcliffe
                return NSE(revAtibainha.D, Q)
            elif FO == 2:
                # SSQ: Sum of Squares of Deviations
                return SSQ(revAtibainha.D, Q)
            else:
                # RMSE: Root-Mean-Square Error
                return RMSE(revAtibainha.D, Q)

    # Busca por evolução diferencial
    result = differential_evolution(objective, bounds, maxiter=1000)
    # Resultados
    print('Muskingum de montante')
    print('Atibaia para Atibainha:')
    print('Status : %s' % result['message'])
    print('Avaliações realizadas: %d' % result['nfev'])
    # Solução
    solution = result['x']
    evaluation = objective(solution)
    print('Solução: \n'
          'f = ( \n'
          '\t[K = %.3f \n\t X = %.3f \n\t m = %.3f]'
          % (solution[0], solution[1], solution[2]))
    if FO == 1:
        print(') = %.3f\n' % (1 - evaluation))
    else:
        print(') = %.3f\n' % evaluation)

    # Armazenamento para plotagem
    upAA = UpstreamFORK(
        solution[0], solution[1], solution[2], 24.0, list(np.multiply(reserv, 0.5)))

    # Armazenamento em dicionário para utilização durante etapa de previsão
    paramsAtibaia['K1'] += [solution[0]]
    paramsAtibaia['X1'] += [solution[1]]
    paramsAtibaia['m1'] += [solution[2]]

    # Mesmo procedimento, porém para Cachoeira e com beta
    beta = 0.5

    # Condições de contorno para variáveis que serão calibradas
    bounds = [
        [120.0, 180.0],         # K (entre 5 e 7.5 dias)
        # X (necessário controlar limite inferior para que o modelo não execute potenciação complexa)
        [0.4,   0.5],
        [1.1,   1.3]          # m (forçando o modelo a não escolher m = 1)
    ]

    # Função objetivo
    def objective(p):
        # Sujeitos a calibração
        K, X, m = p

        # Routing de montante não linear: de Atibaia para Cachoeira
        Q = UpstreamFORK(K, X, m, 24.0, list(np.multiply(reserv, beta)))

        # Restrição positiva ao routing calculado
        res = min(Q)
        if res < 0:
            return np.inf
        else:
            # Métrica utilizada para otimização
            if FO == 1:
                # NSE: Nash-Sutcliffe
                return NSE(revCachoeira.D, Q)
            elif FO == 2:
                # SSQ: Sum of Squares of Deviations
                return SSQ(revCachoeira.D, Q)
            else:
                # RMSE: Root-Mean-Square Error
                return RMSE(revCachoeira.D, Q)

    # Busca por evolução diferencial
    result = differential_evolution(objective, bounds, maxiter=1000)
    # Resultados
    print('Muskingum de montante')
    print('Atibaia para Cachoeira:')
    print('Status : %s' % result['message'])
    print('Avaliações realizadas: %d' % result['nfev'])
    # Solução
    solution = result['x']
    evaluation = objective(solution)
    print('Solução: \n'
          'f = ( \n'
          '\t[K = %.3f \n\t X = %.3f \n\t m = %.3f]'
          % (solution[0], solution[1], solution[2]))
    if FO == 1:
        print(') = %.3f\n' % (1 - evaluation))
    else:
        print(') = %.3f\n' % evaluation)

    # Armazenamento para plotagem
    upAC = UpstreamFORK(solution[0], solution[1], solution[2], 24.0, list(
        np.multiply(reserv, beta)))

    # Armazenamento em dicionário para utilização durante etapa de previsão
    paramsAtibaia['K2'] += [solution[0]]
    paramsAtibaia['X2'] += [solution[1]]
    paramsAtibaia['m2'] += [solution[2]]

    # Inc1: via SMAP
    # Inc2: via Muskingum não-linear tradicional
    resultados = pd.DataFrame(data={
        'Dia': obsAtibaia.t,
        'Pluv. de Atibaia': obsAtibaia.P,
        'Pluv. de Valinhos': obsValinhos.P,
        'Inc2 de Atibaia': incAtibaia,
        'Inc1 de Atibaia': calcAtibaia,
        'Inc2 de Valinhos': incValinhos,
        'Inc1 de Valinhos': calcValinhos,
        'Upst. VA': upVA,
        'Upst. AA': upAA,
        'Upst. AC': upAC
    })

    return paramsAtibaia, paramsValinhos, resultados
