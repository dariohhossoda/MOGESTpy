<div align="center">

# MOGESTpy

Hydrological and water resources management modeling library in Python, supporting quantity and quality simulations for decision-making.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![PyPI](https://img.shields.io/pypi/v/MOGESTpy)](https://pypi.org/project/MOGESTpy/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19308873.svg)](https://doi.org/10.5281/zenodo.19308873)

<img src="https://user-images.githubusercontent.com/58784697/210153011-0ccac06f-5ff8-4c80-a2ad-d03528e71e3e.svg" width="200"/>

</div>

---

<details>
<summary><strong>Table of Contents</strong></summary>

- [MOGESTpy](#mogestpy)
  - [What is MOGESTpy?](#what-is-mogestpy)
    - [Water Resources Management Model](#water-resources-management-model)
  - [Models used in MOGESTpy](#models-used-in-mogestpy)
  - [Repository Structure](#repository-structure)
  - [How to Use MOGESTpy?](#how-to-use-mogestpy)
    - [Installing MOGESTpy](#installing-mogestpy)
      - [Locally](#locally)
      - [Using pip](#using-pip)
  - [How to Contribute to the Project?](#how-to-contribute-to-the-project)
  - [Citation](#citation)

</details>

---

## What is MOGESTpy?

MOGESTpy is an acronym for **Mo**delo de **Gest**ão de Recursos Hídricos (Water Resources Management Model), developed in Python.

### Water Resources Management Model

A water resources management model integrates quantity and quality modeling tools to support decision-making processes. These models simulate natural processes, from precipitation over the watershed to flow routing in rivers and channels, ultimately representing the system response at the watershed outlet.

---

## Models used in MOGESTpy

The models currently available in MOGESTpy include:

* [SIHQUAL](mogestpy/quantity/Hydrodynamic/)
* [SMAP](mogestpy/quantity/Hydrological)
* [0D (Zero Dimensional)](mogestpy/quality/)
* [BuildUp-Washoff](mogestpy/quality/)

For more details, please refer to the documentation within the repository.

---

## Repository Structure

The repository is organized into three main components:

* **Quality models** (`Quality`)
* **Quantity models** (`Quantity`)
* **Datasets and simulation examples** (`Datasets`)

```
MOGESTpy
├─ Datasets
│  └─ Results
├─ quality
│  ├─ BuWo.py
│  └─ ZeroD.py
└─ quantity
   ├─ Hydrological
   │  ├─ SMAP.py
   │  ├─ SMAPm.py
   │  ├─ Muskingum.py
   │  └─ MassBalance.py
   └─ Hydrodynamic
      └─ SIHQUAL.py
```

---

## How to Use MOGESTpy?

### Installing MOGESTpy

#### Locally

#### Using pip

The package is available on PyPI, and you can install it using pip:

```bash
pip install mogestpy
```

Alternatively, it is possible to clone the repository and install the package using pip:

```bash
pip install .
```

Or in editable mode:

```bash
pip install -e .
```

---


## How to Contribute to the Project?

Contributions are welcome!

* Open an issue to report bugs or suggest features
* Submit pull requests with improvements or new models

Please make sure to follow the guidelines described in the [contributing document](CONTRIBUTING.md).

---

## Citation

If you use MOGESTpy in your research, please cite:

```bibtex
@software{mogestpy,
  author = {Dario Hachisu Hossoda},
  title = {MOGESTpy},
  year = {2026},
  url = {https://doi.org/10.5281/zenodo.19308873},
  doi = {10.5281/zenodo.19308873}
}
```

---

<div align="center">

|                                                    Acknowledgments                                                   |
| :------------------------------------------------------------------------------------------------------------------: |
| ![LogoLabSid](https://user-images.githubusercontent.com/58784697/200078179-ea05ba48-2b67-4f30-bea0-78e1c1507ae1.svg) |
|                         [Laboratory of Decision Support Systems](https://labsid.poli.usp.br/)                        |

</div>
