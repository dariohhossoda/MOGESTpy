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

$$ Rsolo (1) = Tuin . Str $$

$$ Rsup  (1) =  0$$

$$ Rsub  (1) = Ebin / (1-kk) / Ad * 86.4 $$

Onde:

$Tuin$ = teor de umidade inicial (ad.)

$Ebin$ = vazão básica inicial (m3/s)

$Ad$   = área de drenagem (km2)


<!-- TODO: Finalizar Documentação -->
