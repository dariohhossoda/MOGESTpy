class Muskingum:
    def downstream_routing(upstream, K, x, T):
        """
        Performs downstream routing from upstream to downstream
        using the Muskingum method.
        """
        # Coefficients
        C0 = (T - (2 * K * x)) / ((2 * K * (1 - x)) + T)
        C1 = (T + (2 * K * x)) / ((2 * K * (1 - x)) + T)
        C2 = ((2 * K * (1 - x)) - T) / ((2 * K * (1 - x)) + T)

        n = len(upstream)
        downstream = [0] * n
        # Initial downstream value is equal to upstream value
        downstream[0] = upstream[0]
        # Loop through from the second to the last entries
        for i in range(1, n):
            downstream[i] = C0 * upstream[i] + C1 * upstream[i - 1] + C2 * downstream[i - 1]

        return downstream

    def DownstreamFORK(K, X, m, T, I):
        """
        Modelo não-linear de Muskingum (de primeira ordem)
        com método de Runge-Kutta de quarta ordem (routing
        de montante para jusante). Variáveis K, X e m devem
        ser calibradas. I refere-se a input, ou hidrograma
        de montante, e T ao time step envolvido (neste caso,
        24 horas)
        """
        n = len(I)

        # Outflow
        O = [0] * n
        # Valor inicial
        O[0] = I[0]
        # Armazenamento
        S = [0] * n

        for i in range(n - 1):
            # Armazenamento atual
            S[i] = K * (X * I[i] + (1 - X) * O[i]) ** m
            # Coeficientes
            k1 = (-1 / (1 - X)) * ((S[i] / K) ** (1 / m) - I[i])
            k2 = (-1 / (1 - X)) * (((S[i] + 0.5 * T * k1) / K) ** (1 / m) - 0.5 * (I[i] + I[i + 1]))
            k3 = (-1 / (1 - X)) * (((S[i] + 0.5 * T * k2) / K) ** (1 / m) - 0.5 * (I[i] + I[i + 1]))
            k4 = (-1 / (1 - X)) * (((S[i] + 1.0 * T * k3) / K) ** (1 / m) - I[i + 1])
            # Armazenamento seguinte
            S[i + 1] = S[i] + T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Outflow seguinte
            O[i + 1] = (1 / (1 - X)) * ((S[i + 1] / K) ** (1 / m) - X * I[i + 1])

        return O

    def UpstreamFORK(K, X, m, T, O):
        """
        Modelo nao-linear de Muskingum (de primeira ordem)
        com metodo de Runge-Kutta de quarta ordem (routing
        de jusante para montante). Variaveis K, X e m devem
        ser calibradas. O refere-se a output, ou hidrograma
        de jusante, e T ao time step envolvido (neste caso,
        24 horas)
        """

        n = len(O)

        # Inflow
        I = [0] * n
        # Valor inicial
        I[n - 1] = O[n - 1]

        # Armazenamento, taxa de variacao
        S = [0] * n

        for i in range(n - 1, 0, -1):
            # Armazenamento
            S[i] = K * (X * I[i] + (1 - X) * O[i]) ** m
            # Coeficientes
            k1 = (1 / X) * ((S[i] / K) ** (1 / m) - O[i])
            k2 = (1 / X) * (((S[i] + 0.5 * T * k1) / K) ** (1 / m) - (0.5 * (O[i] + O[i - 1])))
            k3 = (1 / X) * (((S[i] + 0.5 * T * k2) / K) ** (1 / m) - (0.5 * (O[i] + O[i - 1])))
            k4 = (1 / X) * (((S[i] + 1.0 * T * k3) / K) ** (1 / m) - O[i - 1])
            # Armazenamento em t - 1 (passo anterior)
            S[i - 1] = S[i] - T * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Inflow em t - 1 (passo anterior)
            I[i - 1] = (1 / X) * ((S[i - 1] / K) ** (1 / m)) - ((1 - X) / X) * O[i - 1]

        return I
