Welcome to 2D Heat Equation Solver's documentation!
================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Introduction
-----------
This package provides a numerical solver for the 2D Heat Equation with customizable parameters
and visualization capabilities.

Installation
------------
To install the package, clone the repository and install the requirements:

.. code-block:: bash

   git clone https://github.com/Zvapo/HeatEquation.git
   cd heat-equation-solver
   pip install -r requirements.txt

Usage
-----
You can run the solver using the command line interface:

.. code-block:: bash

   python main.py --help

For example:

.. code-block:: bash

   python main.py --lx 2.0 --ly 2.0 --dx 0.05 --kappa 0.5 --dt 0.0005 --n-steps 2000

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 