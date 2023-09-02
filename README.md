<div id="top"></div>

<div align=center>
<!-- https://user-images.githubusercontent.com/58784697/200683473-b94e7a80-6f62-405d-8ba9-06ac5620044e.svg -->

# MOGESTpy

<p>
  <img src="https://user-images.githubusercontent.com/58784697/210153011-0ccac06f-5ff8-4c80-a2ad-d03528e71e3e.svg"
  width = 200/>
</p>
</div>

<details>
<Summary><font size="5">Table of Contents</font></Summary>

- [MOGESTpy](#mogestpy)
  - [What is MOGESTpy?](#what-is-mogestpy)
    - [Water Resources Management Model](#water-resources-management-model)
  - [Models used in MOGESTpy](#models-used-in-mogestpy)
  - [Repository Structure](#repository-structure)
- [MOGESTpy](#mogestpy-1)
  - [What is MOGESTpy?](#what-is-mogestpy-1)
    - [Water Resources Management Model](#water-resources-management-model-1)
  - [Models used in MOGESTpy](#models-used-in-mogestpy-1)
  - [Repository Structure](#repository-structure-1)
  - [How to Use MOGESTpy?](#how-to-use-mogestpy)
  - [How to Contribute to the Project?](#how-to-contribute-to-the-project)

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

## What is MOGESTpy?

MOGESTpy is an acronym for **Mo**del of **Gest**ion of Water Resources developed in the [Python3](https://www.python.org/) programming language.

### Water Resources Management Model

A management model uses water quantity and quality modeling tools to assist in decision-making. This modeling is done through simulation models where natural processes are represented from precipitation in the watershed area to its transport in rivers and channels to the watershed's outlet.

<!-- Thesis Figure BRITES -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/58784697/199972694-102a218c-bf48-4db4-b0ab-e96ff2609ebd.svg" />
</p>

Tools for the Water Resources Management Process[^Brites,2010]

## Models used in MOGESTpy

The models used in MOGESTpy include:

- [SIHQUAL](Quantity/Hydrodynamic/)
- [SMAP](Quantity/Hydrological)
- [0D (Zero Dimensional)](Quality/)
- [BuildUp-Washoff](Quality/)

For more details, check the documentation within the repository.

<p align="right">(<a href="#top">back to top</a>)</p>

## Repository Structure

The repository is divided into three main parts: quality models (Quality), quantitative models (Quantity), and simulation data for testing and examples of how each module works (Datasets).



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

<div id="top"></div>

<div align=center>
<!-- https://user-images.githubusercontent.com/58784697/200683473-b94e7a80-6f62-405d-8ba9-06ac5620044e.svg -->

# MOGESTpy

<p>
  <img src="https://user-images.githubusercontent.com/58784697/210153011-0ccac06f-5ff8-4c80-a2ad-d03528e71e3e.svg"
  width = 200/>
</p>
</div>

<details>
<Summary><font size="5">Table of Contents</font></Summary>

- [MOGESTpy](#mogestpy)
  - [What is MOGESTpy?](#what-is-mogestpy)
    - [Water Resources Management Model](#water-resources-management-model)
  - [Models used in MOGESTpy](#models-used-in-mogestpy)
  - [Repository Structure](#repository-structure)
- [MOGESTpy](#mogestpy-1)
  - [What is MOGESTpy?](#what-is-mogestpy-1)
    - [Water Resources Management Model](#water-resources-management-model-1)
  - [Models used in MOGESTpy](#models-used-in-mogestpy-1)
  - [Repository Structure](#repository-structure-1)
  - [How to Use MOGESTpy?](#how-to-use-mogestpy)
  - [How to Contribute to the Project?](#how-to-contribute-to-the-project)

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

## What is MOGESTpy?

MOGESTpy is an acronym for **Mo**del of **Gest**ion of Water Resources developed in the [Python3](https://www.python.org/) programming language.

### Water Resources Management Model

A management model uses water quantity and quality modeling tools to assist in decision-making. This modeling is done through simulation models where natural processes are represented from precipitation in the watershed area to its transport in rivers and channels to the watershed's outlet.

<!-- Thesis Figure BRITES -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/58784697/199972694-102a218c-bf48-4db4-b0ab-e96ff2609ebd.svg" />
</p>

Tools for the Water Resources Management Process[^Brites,2010]

## Models used in MOGESTpy

The models used in MOGESTpy include:

- [SIHQUAL](Quantity/Hydrodynamic/)
- [SMAP](Quantity/Hydrological)
- [0D (Zero Dimensional)](Quality/)
- [BuildUp-Washoff](Quality/)

For more details, check the documentation within the repository.

<p align="right">(<a href="#top">back to top</a>)</p>

## Repository Structure

The repository is divided into three main parts: quality models (Quality), quantitative models (Quantity), and simulation data for testing and examples of how each module works (Datasets).

```
MOGESTpy
├─ Datasets
│ └─ Results
├─ Quality
│ ├─ BuWo.py
│ └─ ZeroD.py
└─ Quantity
├─ Hydrological
│ ├─ SMAP.py
│ ├─ Muskingum.py
│ └─ MassBalance.py
└─ Hydrodynamic
└─ SIHQUAL.py
```

<p align="right">(<a href="#top">back to top</a>)</p>

## How to Use MOGESTpy?

Currently, to run MOGESTpy, you need to clone the repository to your machine and call the modules individually. In the future, a framework is planned to be included to facilitate its use.

<p align="right">(<a href="#top">back to top</a>)</p>

## How to Contribute to the Project?

To contribute to the project, simply create an issue. Code additions are welcome; just follow our [contributing document](CONTRIBUTING.md).

<p align="right">(<a href="#top">back to top</a>)</p>

<div align=center>

|Acknowledgments|
|:---:|
|![LogoLabSid](https://user-images.githubusercontent.com/58784697/200078179-ea05ba48-2b67-4f30-bea0-78e1c1507ae1.svg)|
|[Laboratory of Decision Support Systems](https://sites.usp.br/labsid)|

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

[^Brites,2010]: Brites, A. P. Z. Enquadramento dos corpos de água através de metas progressivas: Probabilidade de ocorrência e custos de despoluição hídrica. Tese de Doutorado. Universidade de São Paulo, 2010.
