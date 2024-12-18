# A Python Module for the Heat Equation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of the 2D heat equation for simulating temperature diffusion with optional heat sources.

![Heat equation simulation](images/2d-heat-discretized.gif)

## Table of Contents
- [Overview](#overview)
- [Sources](#sources)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Overview
This is a Python module developed during MSc studies. It is a simple implementation of the heat equation in 2D.
It can be used to simulate the spread of heat in a 2D domain, with two different types of optional sources of heat.

### Sources
 
Type 1 holds a location at a fixed value,

$$ \tag{type 1} T[I_s, J_s] = K,$$

Type 2 applies an additional fixed forcing of K

$$ \tag{type 2} T[I_s, J_s] = T[I_s, J_s] + K $$

## Installation

```bash
pip install -r requirements.txt
```
```bash
conda env create -f environment.yml
```

or 

```bash
pip install -r requirements.txt
```

## Dependencies
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

*The easiest way to install the dependencies is to use the requirements.txt file or create a new environment with the dependencies using the environment.yml file.*

## Usage
The heat equation solver can be used as follows:

```python
from heat_equation import State, Model, Source

# Initialize the environment
state = State(
    initial_value=30.0,
    boundary_value=10.0,
    lx=1.0,
    ly=1.0,
    dx=0.1,
    dtype=np.float64
)

# Optional: Add sources
sources = [
    Source(i_index=5, j_index=5, value=50.0, source_type=1),
    Source(i_index=2, j_index=2, value=50.0, source_type=2)
]

# Initialize the model
model = Model(
    state=state,
    kappa=0.1,
    sources=sources
)

# run the model
model.run(
    dt=0.01,
    n_steps=5,
    filename="run_1",
    plot_steps=1,
    show=True
)
```

*The result of this implementation is shown on the gif attached at the top of the README.*

or use the command line interface:

```bash
python main.py --help
```

```bash
python main.py --lx 1.0 --ly 1.0 --dx 0.1 --kappa 0.1 --dt 0.0005 --n-steps 100
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.







