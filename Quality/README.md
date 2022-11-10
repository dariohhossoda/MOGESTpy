# Módulos de Qualidade - MOGESTpy

- [Módulos de Qualidade - MOGESTpy](#módulos-de-qualidade---mogestpy)
  - [Modelo BuildUp-Washoff (BuWo)](#modelo-buildup-washoff-buwo)
  - [Modelo 0D (Zero-Dimensional)](#modelo-0d-zero-dimensional)

## Modelo BuildUp-Washoff (BuWo)

## Modelo 0D (Zero-Dimensional)

O modelo 0D é derivado da modelagem de reatores de mistura contínua, sendo um dos sistemas mais simples a ser modelado [^Chapra,2008]. Sendo o modelo proposto derivado da equação de balanço de massa:

$$\text{Accumulation} = \text{Loading} - \text{Outflow} - \text{Reaction} - \text{Settling} $$

O modelo então é resolvido através do método das diferenças finitas, assim como feito por Becker (2021) [^Becker,2021], com o seguinte equacionamento:

$$ C_{out}^{n+1} = \frac{C_{out}^n + \frac{\Delta t}{V^{n+1}}(Q_{in}^{n+1} C_{in}^{n+1})}{1+\frac{\Delta t}{V^{n+1}}[Q_{out}^{n+1}+kV^{n+1}+vA_s^{n+1}+\frac{\Delta V}{\Delta t}]}$$

Onde:

- $C_{out}$ é a concentração na saída em mg/L;
- $C_{in}$ é a concentração na entrada em mg/L;
- $\Delta t$ é a diferença entre os tempos $i+1$ e $i$ em s;
- $V$ é o volume do reservatório em m $^3$;
- $\Delta V$ é a diferença dos volumes a cada instante em m $^3$;
- $v$ é a velocidade aparente de sedimentação em m/s;
- $k$ é o coeficiente de reação de primeira ordem s $^{-1}$;
- $A_s$ é a área de superfície entre água e sedimentos em m $^2$.



[^Becker,2021]: Becker, A. C. C. Zero-Dimensional Modelling and Total Maximum Daily Loads As Tools for Reservoir Water Quality Planning and Management. Master Thesis. Universidade Federal do Paraná - UFPR, 2021.

[^Chapra,2008]: Chapra, S. C. Surface Water-Quality Modeling. Waveland press, 2008.
