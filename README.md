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
  - [How to Use MOGESTpy?](#how-to-use-mogestpy)
    - [Installing MOGESTpy](#installing-mogestpy)
  - [How to Contribute to the Project?](#how-to-contribute-to-the-project)

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

## What is MOGESTpy?

MOGESTpy is an acronym for **Mo**delo de **Gest**ão de Recursos Hídricos (in English: Water Resources Management Model) developed in the [Python3](https://www.python.org/) programming language.

### Water Resources Management Model

A management model uses water quantity and quality modeling tools to assist in decision-making. This modeling is done through simulation models where natural processes are represented from precipitation in the watershed area to its transport in rivers and channels to the watershed's outlet.

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

### Installing MOGESTpy

Using the following command, you can install MOGESTpy via pip:

```bash
pip install mogestpy
```

Alternatively, you can clone the repository and install the package locally:

```bash
pip install .
```

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
