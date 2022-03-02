# synthesis_based_repair

## Setup

### Installing packages

Other repositories:

  git clone git@github.com:craigiedon/dl2_lfd.git

Currently, dl2_lfd is hardcoded to use cuda instead of cpu.
If you want to use cpu instead of cuda, you will need to change mentions of "cuda" in dl2_lfd/dmps/dmp.py to "cpu".

Required packages:

- numpy
- torch
- scipy
- matplotlib

Add to path:
  export PYTHONPATH=$PYTHONPATH:~/repos:/~/repos/synthesis_based_repair:~/repos/synthesis_based_repair/src

## Symbols

### Defining Symbols

To define symbols, write a .json file that defines each symbol.
Example json file for y2:

  {
    "y2": {"dims": [0, 1],
          "name": "y2",
          "bounds": [[0, 3], [2, 3]],
          "index": 5,
          "factor": 1,
          "color": "orange",
          "type": "rectangle"}
        }
    }

Type can be either "rectangle" or "circle".

### Visualizing Symbols

The symbol class (synthesis_based_repair/src/symbols.py) has a function that plots the symbol.
To generate a sample plot for the Nine Squares example, from /synthesis_based_repair/src, run:

  python symbols.py

## Skills

### Generate Trajectories

To generate trajectory data for the Nine Squares example, from `synthesis_based_repair` run:

  cd scripts
  python generate_trajectories.py

If you want the code to run faster, you will want to comment out Line 8 of synthesis_based_repair/ltl_diff/constraints.py and add:

  neg_losses = torch.zeros(sat.shape)

to Line 11 of synthesis_based_repair/ltl_diff/constraints.py.

### Visualizing Trajectories

The skills class (synthesis_based_repair/src/skills.py) has a trajectory plotting function.
Symbols can also be visualized under the trajectory.
To see an example, from `synthesis_based_repair/src` run:

  python skills.py

## Specification

After generating symbols and trajectories, you can generate the specification that encodes the trajectories and task.
The `tools.py` file contains `write_spec` that takes the user specification, task, and symbols, and creates a structuredslugs file.
The user specification is passed in the form of a json file.
For the nine squares example, see `synthesis_based_repair/data/nine_squares/nine_squares_a.json` (duplicated here)

  {
    "env_init_true": ["x0", "y0"],
    "env_init_false": ["x1", "x2", "y1", "y2"],
    "sys_init_true": [],
    "sys_init_false": ["skill0", "skill1", "skill0b", "skill1b"],
    "sys_live": "x0 & y0'\nx2 & y2'\n",
    "env_live": "",
    "sys_trans_hard": "!(x2 & y0)'\n!(x2' & y0')",
    "reactive_variables": [],
    "change_cons": "",
    "not_allowed_repair": "",
    "hard_constraints": [{"x2": "True", "y0": "True"}]
  }

To generate the specification for the Nine Squares example, from `synthesis_based_repair\src\` run:

  python test_specification.py

## Synthesis-Based Repair without Physical Feedback

The synthesis-based repair algorithm is in `run_repair` in `synthesis_based_repair\src\symbolic_repair`.
To run the Nine Squares example, from `synthesis_based_repair\src`, run:

  python test_symbolic_repair.py

## Synthesis-Based Repair with Physical Feedback

The physical portion of the repair algorithm is in `run_elaborateDMP` in `synthesis_based_repair\src\physical_implementation`.
To run the Nine Squares example, from `synthesis_based_repair\src`, run:

  python test_symbolic_physical_integration_repair.py

Note that this will use the DMPs and may take some time.

## Running examples

## Running your own example

## Notes
