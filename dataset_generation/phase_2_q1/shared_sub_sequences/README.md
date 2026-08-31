## How to generate the synthetic datasets used in the Phase-2 for Q1 that assume shared sequence patterns constitute immune signals

- This folder contains all the files required to generate the synthetic datasets used in the Phase-2 for Q1, where shared sequence patterns are assumed to constitute immune signals. The synthetic datasets are generated using the following console script, assuming this python package is installed:

```bash
nohup multi_ligo -i relative_freqs_filepaths.yaml >> stdout.txt 2>> stderr.txt &
```

- To convert the LIgO-generated output to the desired uniform format that will be supplied to the participants, the following command can be used on each of the generated datasets. As an example:

```bash
process_ligo_output -l /path/to/q1_kmers_data/q1_kmers/run1/phenotype_burden_5/data
```

- Note that LIgO version 1.0.2 has to be installed in the conda environment or similar for the above command to work.

- The above command generates simulations specification files for LIgO and runs the simulations. As an example, the simulation specification files used to generate the datasets used in the Phase-2 for Q1 are shown in the `simulation_scripts` folder. Note that the paths have to be replaced with appropriate paths. There might be slight mismatch between filenames because of typos, but should be straightforward to relate to.