Changelog
=========

Releases are listed on `GitHub <https://github.com/dariohhossoda/MOGESTpy/releases>`_
and published on `PyPI <https://pypi.org/project/mogestpy/>`_, each with a
DOI on Zenodo.

2.1.1 (unreleased)
------------------

* Changelog: release date of 2.1.0 (#78).

2.1.0 (2026-10-12)
------------------

* **Requires Python 3.11 or later and NumPy 2.4.4 or later** (2.0.0 supported
  Python 3.9). Users on older Python versions keep getting 2.0.0 from pip.
* Basin networks: ``run_network`` simulates SMAP per subcatchment with
  Muskingum routing between them, with pass-through reaches (``k = 0``),
  external inflows, custom routing functions and the local (natural) flow as
  output (#50, #52).
* ``MuskingumCunge``: Muskingum-Cunge routing with parameters from the reach
  characteristics, also usable as the ``router`` of ``run_network`` (#53).
* ``SmapD.fast_calibrate``, as in ``SmapM`` (#51).
* ``SmapD.calibrate`` no longer emits deprecation warnings (#58).
* Documentation: Sphinx site on Read the Docs, built from the source, with a
  user guide, executed examples and per-module API reference (#59, #64).
* The citation uses the Zenodo concept DOI, valid for all versions (#55).

2.0.0 (2026-04-16)
------------------

* Package and module names follow the usual Python casing.
* Daily and monthly SMAP models in ``smap.py``.

1.2.1 (2026-03-29)
------------------

* Code cleanup.

1.2.0 (2026-03-25)
------------------

* Compatibility of project versions across platforms.

0.2.0 (2026-03-25)
------------------

* Monthly SMAP model.

0.1.1 (2025-11-05)
------------------

* Fix optional imports of the PSO calibration.

0.1.0 (2025-04-12)
------------------

* New SMAP methods for advanced users.

0.0.2 (2024-10-24)
------------------

* Minor improvements and automated publishing to PyPI.

0.0.1-alpha (2024-06-22)
------------------------

* First release, with models for water resources planning.
