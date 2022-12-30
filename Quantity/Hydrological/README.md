# Modelos Hidrológicos
Esta seção destina-se aos modelos hidrológicos de quantidade de água.

## SMAP

### Introdução

O modelo SMAP (Soil Moisture Accounting Procedure) é um modelo determinístico de simulação hidrológica do tipo transformação chuva-vazão desenvolvido em 1981 por Lopes J.E.G., Braga B.P.F. e Conejo J.G.L., e apresentado no International Symposium on Rainfall-Runoff Modeling realizado em Mississippi, U.S.A. e publicado pela Water Resourses Publications (1982).

O desenvolvimento do modelo baseou-se na experiência com a aplicação do modelo Stanford Watershed IV e modelo Mero em trabalhos realizados no DAEE- Departamento de Águas e Energia Elétrica do Estado de São Paulo. Foi originalmente desenvolvido para intervalo de tempo diário e posteriormente apresentadas  versões horária e mensal, adaptando-se algumas modificações em sua estrutura.

### Metodologia do Modelo

Em sua versão diária, é constituído de três reservatórios matemáticos, cujas variáveis de estado são atualizadas a cada dia da forma:

$$Rsolo (i+1) = Rsolo (i) + P - Es - Er - Rec$$
$$Rsup  (i+1) = Rsup  (i) + Es - Ed$$
$$Rsub  (i+1) = Rsub  (i) + Rec - Eb$$

Onde:

$Rsolo$ = reservatório do solo (zona aerada)

$Rsup$  = reservatório da superfície da bacia

$Rsub$  = reservatório subterrâneo (zona saturada)

$P$     = chuva

$Es$    = escoamento superficial

$Ed$    = escoamento direto

$Er$    = evapotranspiração real

$Rec$   = recarga subterrânea

$Eb$    = escoamento básico

Inicialização:

$$ Rsolo (1) = Tuin \cdot Str $$

$$ Rsup  (1) =  0$$

$$ Rsub  (1) = Ebin / (1-kk) / Ad \cdot 86.4 $$

Onde:

$Tuin$ = teor de umidade inicial ( - )

$Ebin$ = vazão básica inicial (m³/s)

$Ad$   = área de drenagem (km²)


<!-- TODO: Finalizar Documentação -->
<!-- Adicionar Figura -->


O modelo apresenta 5 funções de transferência entre os reservatórios. A separação do escoamento superficial é baseado no método do SCS ( Soil Conservation Service do U.S.Dept. Agr.).

1. Se ($P > Ai$)

Então

$S = Str - Rsolo$

$Es = (P - Ai) ^ 2 / (P - Ai + S)$

Caso contrário

$Es = 0$

2. Se $(P - Es) > Ep$

Então
		
$Er = Ep$

Caso contrário

$Er = (P - Es) + (Ep - (P - Es)) \cdot Tu$

3. Se $Rsolo > (Capc \cdot Str)$

Então

$Rec = Crec \cdot Tu \cdot (Rsolo - (Capc \cdot Str))$

Caso contrário

$Rec = 0$

1. $Ed  = Rsup \cdot ( 1 - K2 )$

2. $Eb  = Rsub \cdot ( 1 - Kk )$


sendo	$Tu = Rsolo / Str$


São 6 os parâmetros do modelo:

$Str$	- capacidade de saturação do solo (mm)

$K2t$	- constante de recessão do escoamento 
superficial (dias)
$Crec$	- parâmetro de recarga subterrânea (%)

$Ai$	- abstração inicial (mm)

$Capc$	- capacidade de campo (%)

$Kkt$	- constante de recessão do escoamento básico (dias)

Foram ajustadas as unidades dos parâmetros:

$Kk = 0,5 ^ {(1/Kkt)}$  e  $K2 = 0,5 ^ {(1/K2t)}$  onde $Kkt$  e  $K2t$  são expressos em dias em que a vazão cai a metade de seu valor.

$Crec$  e  $Capc$  são multiplicados por $100$

O eventual transbordo do reservatório do solo é transformado em escoamento superficial.

Finalmente o cálculo da vazão é dado pela equação:

$$Q = (Es + Eb) * Ad / 86.4$$

Os dados de entrada do modelo são os totais diários de chuva e o total diário médio do período de evaporação potencial (tanque classe A). Para calibração são necessários de 30 a 90 dias de dados de vazão media mensal, incluindo eventos de cheia.

É utilizado um coeficiente de ajuste da chuva media da bacia "$Pcof$" que deve ser calculado em função da distribuição espacial dos postos.
