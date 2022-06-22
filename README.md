# synthesis_based_repair

## Overview

This repository provides the code associated with [Physically-Feasible Repair of Reactive, Linear Temporal Logic-based, High-Level Tasks](https://www.overleaf.com/project/5ddc44589d7d75000192b562).

## Setup

This code requires python > 3 to run and has been tested on Ubuntu 18.04 and 20.04.

### Installing Required Software

#### Other repositories:

This [fork](https://github.com/apacheck/dl2_lfd) of the repository [dl2_lfd](https://github.com/craigiedon/dl2_lfd) must be installed and added to PYTHONPATH.

```shell
git clone --recurse-submodules https://github.com/apacheck/dl2_lfd.git
export PYTHONPATH=$PYTHONPATH:[PARENT DIRECTORY OF dl2_lfd]
```

<!-- Currently, the dl2_lfd repository is hardcoded to use cuda instead of cpu.
If you want to use cpu instead of cuda, you will need to change mentions of "cuda" in dl2_lfd/dmps/dmp.py to "cpu". -->

We use the dd package from tulip-control to handle the BDDs (<https://github.com/tulip-control/dd>).
Full instructions are found at the repository, but this may suffice:

```shell
git clone https://github.com/tulip-control/dd.git
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

```shell
pip3 install numpy torch scipy matplotlib pydo astutils
```

#### Install

Clone and build the repository:

```shell
git clone https://github.com/apacheck/synthesis_based_repair.git
cd synthesis_based_repair
python setup.py install
```

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

The symbol class (`[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/synthesis_based_repair/symbols.py`) has a function that plots the symbol.
To generate a sample plot for the Nine Squares example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts`, run:

```shell
python run_plot_symbols.py --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json" --dmp_opts "../data/nine_squares/nine_squares_dmp_opts.json"
```

A plot of the propositions will be generated in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/plots`.

### Skills

#### Generate Trajectories

To generate trajectory data for the Nine Squares example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts` run:

```shell
python run_generate_trajectories.py --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json" --dmp_opts "../data/nine_squares/nine_squares_dmp_opts.json" --do_plot
```

By default the data for the trajectories will be saved in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/trajectories/`.
The dynamic motion primitive associated with the skills will be save in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/dmps/`.
The .json file associate with the skills that is used when creating the specification will be saved in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/nine_squares_skills.json`.
With used of the flag "--do_plot", a plot of each skill will be saved to `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/plots/[SKILL_NAME].png`.

#### Visualizing Trajectories

The skills class (`[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/synthesis_based_repair/skills.py`) has a trajectory plotting function.
Symbols can also be visualized under the trajectory without generating new data if desired.
To see an example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts` run:

```shell
python run_plot_skills.py --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json" --dmp_opts "../data/nine_squares/nine_squares_dmp_opts.json"
```

A plot of a sample of the trajectories associated with the skills will be generated in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/plots/[SKILL_NAME].png`.

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

The syntax for the initial conditions is to list all the propositions that must be true and must be false.
This will be transformed into the approriate specification in the structuredslugs file.
For the system liveness guarantess, write out the guarantees as it would appear in a structured slugs file.
In the example above, the system liveness is interpretted as "Always eventually x2 and y0 AND Always eventually x2 and y2".

Any user defined hard constraints should be written in "sys_trans_hard".
These will be added to [SYS_TRANS_HARD] in the structuredslugs file.
**NOTE:** If the robot should always avoid a certain combination of propositions, the specification should be written such that it avoids the propositions at both the current and next step.

The change constraints in this example constrains that the propositions are mutually exclusive.
One can add further constraints about which propositions can change to other propositions if desired.

The not_allowed_repair should be initialized to specificy that the repair process cannot change the duplicate skills.

To generate the specification for the Nine Squares example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts` run:

```shell
python run_create_specification.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json"
```

The specification will be in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/nine_squares/nine_squares_a.structuredslugs`.

### Synthesis-Based Repair without Physical Feedback

The synthesis-based repair algorithm is in `run_repair` in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/synthesis_based_repair/symbolic_repair`.
To run the Nine Squares example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts`, run:

```shell
python run_symbolic_repair.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json"
```

The results of the repair process will be displayed to the screen.

### Synthesis-Based Repair with Physical Feedback

The physical portion of the repair algorithm is in `run_elaborateDMP` in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/synthesis_based_repair/physical_implementation`.
To run the Nine Squares example, from `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/scripts`, run:

```shell
python run_symbolic_physical_integration_repair.py --user_spec "../data/nine_squares/nine_squares_a.json" --file_names "../data/nine_squares/nine_squares_files.json" --sym_opts "../data/nine_squares/nine_squares_sym_opts.json" --dmp_opts "../data/nine_squares/nine_squares_dmp_opts.json" --loss_threshold 0.8
```

Note that this will use the DMPs and may take some time.
The results of the repair will be displayed to the screen.
Plot of the intermediate modifications of the repair can be seen in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/logs/`.
The final new dmps will be saved in `[PARENT DIRECTORY TO SYNTHESIS_BASED_REPAIR]/synthesis_based_repair/data/dmps/`.

### Creating your own example

To create your own example, you will need:

1. Symbols
2. Trajectories (and DMPs)
3. Specification

in the formats described above.

#### Symbols
To create the symbols, you can create a .json file by hand.

#### Trajectories/DMPs
To create the trajectories you can modify the "run_generate_trajectories.py" script by adding a function to generate trajectories for your specific case.
You will also need to construct files similar to: nine_squares_files.json, nine_squares_dmp_opts.json, and nine_squares_sym_opts.json.
For more details on these files see below.

#### Specification
Create a file similar to nine_squares_a.json that contains the user provided aspects of the specification.
The parts of the specification relating to the skills will be automatically added with "run_generate_specification.py".

#### Files containing options/parameters

##### [EXAMPLE]_files.json

##### [EXAMPLE]_dmp_opts.json

##### [EXAMPLE]_sym_opts.json

## Notes

### Specification Language

The chosen format for the specifications is structuredslugs.
This means specifications are compatible with [Slugs](https://github.com/VerifiableRobotics/slugs).
However, the [SYS_TRANS_HARD], [ENV_TRANS_HARD], [CHANGE_CONS], and [NOT_ALLOWED_REPAIR] are not allowed in Slugs.
This [fork](https://github.com/apacheck/slugs) allows Slugs to handle it the specifications with these headings.

#### Stretch Example

### Symbols for the Stretch

Symbols are defined in `stretch_symbols.json`.
There are 4 regions.
There are separate symbols for the base and end-effector in each of these regions.

To plot these regions, from `\synthesis_based_repair` run:
```
cd scripts
python run_plot_symbols.py --file_names "../data/stretch/stretch_files.json" --sym_opts "../data/stretch/stretch_sym_opts.json" --dmp_opts "../data/stretch/stretch_dmp_opts.json"
```

Plots will be generated in `synthesis_based_repair/data/stretch/plots/`.
