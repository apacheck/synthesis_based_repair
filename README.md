# synthesis_based_repair

## Overview

This repository provides the code associated with [Physically-Feasible Repair of Reactive, Linear Temporal Logic-based, High-Level Tasks](https://www.overleaf.com/project/5ddc44589d7d75000192b562).

## Setup

### Installing Required Software

#### Other repositories:

This [fork](https://github.com/apacheck/dl2_lfd) of the repository [dl2_lfd](https://github.com/craigiedon/dl2_lfd) must be installed and added to PYTHONPATH.

```shell
git clone --recurse-submodules git@github.com:apacheck/dl2_lfd.git
```

<!-- Currently, the dl2_lfd repository is hardcoded to use cuda instead of cpu.
If you want to use cpu instead of cuda, you will need to change mentions of "cuda" in dl2_lfd/dmps/dmp.py to "cpu". -->

We use the dd package from tulip-control to handle the BDDs (<https://github.com/tulip-control/dd>).
Full instructions are found at the repository, but this may suffice:

```shell
git clone git@github.com:tulip-control/dd
cd dd
pip install cython
python setup.py install --fetch --cudd
```

#### Required python packages:

- numpy
- torch
- scipy
- matplotlib
- pydot
- astutils

#### Install

Clone and build the repository:

```shell
git@github.com:apacheck/synthesis_based_repair.git
cd synthesis_based_repair
python setup.py install
```

<!-- Add the folder to the PYTHONPATH:

```shell
export PYTHONPATH=$PYTHONPATH:[PATH_TO_SYNTHESIS_BASED_REPAIR]
```
 -->
## Usage

### Propositions

#### Defining Propositions

Each proposition grounds to a region of the state space.
In order to use the repair process, write a .json file that defines each symbol.
Example part of a .json file for y2 in Fig. 1 of the paper:

```json
{
  "y2": {"dims": [0, 1],
        "name": "y2",
        "bounds": [[0, 3], [2, 3]],
        "index": 5,
        "factor": 1,
        "color": "orange",
        "type": "rectangle"}
}
```

"dims" is the dimensions the proposition grounds to, "bounds" are the lower and upper limits for the proposition, "index" just keeps a count of the propositions, "factor" defines the different factors if propositions only ground to some dimensions of the state space, "color" is the shading for visualization.
Type can be either "rectangle" or "circle".

#### Visualizing Propositions

The symbol class (`synthesis_based_repair/synthesis_based_repair/symbols.py`) has a function that plots the symbol.
To generate a sample plot for the Nine Squares example, from /synthesis_based_repair/synthesis_based_repair, run:

```shell
python symbols.py
```

### Skills

#### Generate Trajectories

To generate trajectory data for the Nine Squares example, from `synthesis_based_repair` run:

```shell
cd scripts
python generate_trajectories.py
```

<!-- If you want the code to run faster, you will want to comment out Line 8 of `dl2_lfd/ltl_diff/constraints.py` and add:

```python
neg_losses = torch.zeros(sat.shape)
``` -->

to Line 11.

#### Visualizing Trajectories

The skills class (`synthesis_based_repair/synthesis_based_repair/skills.py`) has a trajectory plotting function.
Symbols can also be visualized under the trajectory.
To see an example, from `synthesis_based_repair/synthesis_based_repair` run:

```shell
python skills.py
```

### Specification

After generating symbols and trajectories, you can generate the specification that encodes the trajectories and task.
The `tools.py` file contains `write_spec` that takes the user specification, task, and symbols, and creates a structuredslugs file.
The user specification is passed in the form of a json file.
For the nine squares example in the paper, see `synthesis_based_repair/data/nine_squares/nine_squares_a.json` (duplicated here)

```json
{
  "env_init_true": ["x0", "y0"],
  "env_init_false": ["x1", "x2", "y1", "y2"],
  "sys_init_true": [],
  "sys_init_false": ["skill0", "skill1", "skill0b", "skill1b"],
  "sys_live": "x0 & y0\nx2 & y2\n",
  "env_live": "",
  "sys_trans_hard": "!(x2 & y0)\n!(x2' & y0')",
  "reactive_variables": [],
  "hard_constraints": [{"x2": "True", "y0": "True"}],
  "change_cons": "!(x0' & x1')\n!(x0' & x2')\n!(x1' & x2')\n!(y0' & y1')\n!(y0' & y2')\n!(y1' & y2')\n(x0' | x1' | x2')\n(y0' | y1' | y2')\n!(x0 & x1)\n!(x0 & x2)\n!(x1 & x2)\n!(y0 & y1)\n!(y0 & y2)\n!(y1 & y2)\n(x0 | x1 | x2)\n(y0 | y1 | y2)\n(x0 <-> !x0') | (x1 <-> !x1') | (x2 <-> !x2') | ((x0 <-> x0') & (x1 <-> x1') & (x2 <-> x2'))\n(y0 <-> !y0') | (y1 <-> !y1') | (y2 <-> !y2') | ((y0 <-> y0') & (y1 <-> y1') & (y2 <-> y2'))\n!((x0 <-> x0') & (x1 <-> x1') & (x2 <-> x2') & (y0 <-> y0') & (y1 <-> y1') & (y2 <-> y2'))\n",
  "not_allowed_repair": "!skill0b\n!skill1b\n"
}
```

To generate the specification for the Nine Squares example, from `synthesis_based_repair\scripts` run:

```shell
python run_create_specification.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json"
```

### Synthesis-Based Repair without Physical Feedback

The synthesis-based repair algorithm is in `run_repair` in `synthesis_based_repair\synthesis_based_repair\symbolic_repair`.
To run the Nine Squares example, from `synthesis_based_repair\scripts`, run:

```shell
python run_symbolic_repair.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json"
```

### Synthesis-Based Repair with Physical Feedback

The physical portion of the repair algorithm is in `run_elaborateDMP` in `synthesis_based_repair\synthesis_based_repair\physical_implementation`.
To run the Nine Squares example, from `synthesis_based_repair\scripts`, run:

```shell
python run_symbolic_physical_integration_repair.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json" --dmp_opts "../data/nine_squares/nine_squares_dmp_opts.json" --loss_threshold 0.8
```

Note that this will use the DMPs and may take some time.

### Creating your own example

To create your own example, you will need:

1. Symbols
2. Trajectories (and DMPs)
3. Specification

in the formats described above.

## Notes

### Specification Language

The chosen format for the specifications is structuredslugs.
This means specifications are compatible with [Slugs](https://github.com/VerifiableRobotics/slugs).
However, the [SYS_TRANS_HARD], [ENV_TRANS_HARD], [CHANGE_CONS], and [NOT_ALLOWED_REPAIR] are not allowed in Slugs.
This [fork](https://github.com/apacheck/slugs) allows Slugs to handle it the specifications with these headings.
