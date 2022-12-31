<div id="top"></div>

# MOGESTpy

<p align="center">
  <img src="https://user-images.githubusercontent.com/58784697/200683473-b94e7a80-6f62-405d-8ba9-06ac5620044e.svg"
  width = 250/>
</p>

## Índice

- [MOGESTpy](#mogestpy)
  - [Índice](#índice)
  - [O que é o MOGESTpy?](#o-que-é-o-mogestpy)
    - [Modelo de Gestão de Recursos Hídricos](#modelo-de-gestão-de-recursos-hídricos)
  - [Modelos utilizados no MOGESTpy](#modelos-utilizados-no-mogestpy)
  - [Estrutura do Repositório](#estrutura-do-repositório)
  - [Como utilizar o MOGESTpy?](#como-utilizar-o-mogestpy)
  - [Como colaborar com o projeto?](#como-colaborar-com-o-projeto)

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

## O que é o MOGESTpy?

O projeto MOGESTpy é a sigla para **Mo**delo de **Gest**ão de Recursos Hídricos desenvolvido em linguagem Python.

### Modelo de Gestão de Recursos Hídricos

Um modelo de gestão utiliza ferramentas de modelagem de quantidade e qualidade da água como forma de auxiliar a tomada de decisão. Esta modelagem é feita por através de modelos de simulação em que processos naturais são representados desde a precipitação na área da bacia até seu transporte em rios e canais para o exutório da bacia.

<!-- Figura Tese BRITES -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/58784697/199972694-102a218c-bf48-4db4-b0ab-e96ff2609ebd.svg" />
</p>

Ferramentas para o Processo de Gestão de Recursos Hídricos[^Brites,2010]


## Modelos utilizados no MOGESTpy
- [SIHQUAL](Quantity/Hydrodynamic/)
- [SMAP](Quantity/Hydrological)
- [0D (Zero Dimensional)](Quality/)
- [BuildUp-Washoff](Quality/)
<p align="right">(<a href="#top">voltar ao topo</a>)</p>

## Estrutura do Repositório

```
MOGESTpy
├─ Datasets
│   └─ Results
├─ Quality
│   ├─ BuWo.py
│   └─ ZeroD.py
└─ Quantity
    ├─ Hydrological
    │   ├─ SMAP.py
    │   ├─ Muskingum.py
    │   └─ MassBalance.py
    └─ Hydrodynamic
        └─ SIHQUAL.py
```

<p align="right">(<a href="#top">voltar ao topo</a>)</p>

## Como utilizar o MOGESTpy?

Atualmente, para executar o MOGESTpy é necessário clonar o repositório em sua máquina e chamar os módulos individualmente. Futuramente pensa-se em incluir um Framework para facilitar sua utilização.

## Como colaborar com o projeto?

Para contribuir com o projeto basta criar uma issue, adições ao código são bem-vindas, basta seguir o nosso [documento de colaboração](CONTRIBUTING.md).

<div align=center>

|Agradecimentos|
|:---:|
|![LogoLabSid](https://user-images.githubusercontent.com/58784697/200078179-ea05ba48-2b67-4f30-bea0-78e1c1507ae1.svg)|
|[Laboratório de Sistemas de Suporte a Decisões](https://sites.usp.br/labsid)|

</div>

[^Brites,2010]: Brites, A. P. Z. Enquadramento dos corpos de água através de metas progressivas: Probabilidade de ocorrência e custos de despoluição hídrica. Tese de Doutorado. Universidade de São Paulo, 2010.
