# Modelos Hidrodinâmicos

Modelos hidrodinâmicos são modelos matemáticos utilizados para descrever o movimento da água ao longo do tempo em situações diferentes do regime permanente, chamado regime transitório.

## SIHQUAL

O SIHQUAL é o modelo de Simulação Hidrodinâmica e de Qualidade da Água, desenvolvido por Ferreira [^Ferreira,2015], através do acoplamento das equações de Saint-Venant com a equação de Advecção - Dispersão - Difusão de cargas transportadas pelo corpo hídrico.

### Equações de Saint-Venant

As equações de Saint-Venant são um conjunto de equações diferenciais parciais (EDPs) que descrevem o comportamento da água em rios e canais ao longo da dimensão longitudinal (1D), nomeadas a partir dos trabalhos desenvolvidos pelo matemático francês Adhémar Jean Claude Barré de Saint-Venant, que as desenvolveu no século XIX.

As equações são derivadas dos princípios de conservação de massa (continuidade) e de conservação da quantidade de movimento (momento linear).

#### Equação da Continuidade

$$ B \frac{\partial{y}}{\partial{t}} + UB \frac{\partial{y}}{\partial{x}} + A \frac{\partial{U}}{\partial{x}} + U \frac{\partial{A}}{\partial{x}} = q$$

#### Equação da Conservação da Quantidade de Movimento

$$ \frac{\partial{U}}{\partial{t}} + U\frac{\partial{U}}{\partial{x}} + g\frac{\partial{y}}{\partial{x}} = \frac{q(v_L - U)}{A} + g(S_0 - S_f) $$


### Parâmetros de configuração

$\mathrm{d}x$: passo espacial (m)

$\mathrm{d}t$: passo de tempo (s)

$L$: comprimento total do curso d'água (m)

$tf$: tempo total de simulação (s)

$g$: aceleração da gravidade (m$2$/s)

$\alpha$: coeficiente de difusão numérica de Lax

$D$: coeficiente de dispersão

[^Ferreira,2015]: Ferreira, D, M. Simulação Hidrodinâmica e de Qualidade da Água em Rios: Impacto para os Instrumentos de Gestão de Recursos Hídricos. Tese de Mestrado. Universidade Federal do Paraná, 2015.