# Data

Raw data is not committed to Git. Download it and place it as below.

## Scania Component X (main case)

1. Download the dataset from the Swedish National Data Service (open access,
   CC BY 4.0): https://doi.org/10.5878/bnh5-ka77
2. Put these nine CSV files in `data/scania/`:

```text
data/scania/
├── train_operational_readouts.csv        train_specifications.csv        train_tte.csv
├── validation_operational_readouts.csv   validation_specifications.csv   validation_labels.csv
└── test_operational_readouts.csv         test_specifications.csv         test_labels.csv
```

The validation specifications file may be spelled slightly differently; the
loader accepts any `validation_spec*.csv`. The first run caches each table as
parquet in `data/scania/cache/` (also not committed).

## NASA C-MAPSS (benchmark)

1. Download the dataset from NASA's open data portal:
   https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
2. Unzip it (there may be a zip inside the zip).
3. Copy these twelve files into this `data/` folder:

```text
data/
├── train_FD001.txt   test_FD001.txt   RUL_FD001.txt
├── train_FD002.txt   test_FD002.txt   RUL_FD002.txt
├── train_FD003.txt   test_FD003.txt   RUL_FD003.txt
└── train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

To run only some subsets, copy just their files and use
`python -m mfp.pipeline --subset FD001` (or several names).

Both pipelines check that the files exist and are well formed before they run.
