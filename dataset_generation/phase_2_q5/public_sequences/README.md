## How to generate the synthetic datasets used in the Phase-2 for Q5 that assume public sequences constitute immune signals

- This folder contains all the files required to generate the synthetic datasets used in Phase-2 for Q5, where public sequences are assumed to constitute immune signals. The synthetic datasets are generated using the following console script, assuming this python package is installed:

```bash
nohup multi_simairr_var -i q5_fullseq_config.yaml >> stdout.txt 2>> stderr.txt &
```

- Note that the simulation specification file uses a large file in the `background_sequences_path` field. This file is not included in the repository because of its size. The file is available for download at the following DOI: doi.org/10.11582/2024.00176

- In addition, multiple signal sequence files are used, which refer to the `/path/to/vdjdb_downloaded/`. This `vdjdb_downloaded` directory is provided under submodule `preprocess_vdjdb_sequences` in this repository. 

- Note that the latest version of simAIRR has to be installed in the conda environment or similar for the above command to work.

- The above command generates simulations specification files for simAIRR and runs the simulations. As an example, the simulation specification files used to generate the datasets used in the Phase-2 for Q5 are shown in the `simulation_scripts` folder. Note that the paths have to be replaced with appropriate paths. There might be slight mismatch between filenames because of typos, but should be straightforward to relate to.