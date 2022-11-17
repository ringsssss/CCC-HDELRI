# CellComm
CellComm: Integrating potential ligand-receptor interactions and single-cell RAN sequencing data for cell-cell communication inference

## Data
Data is available at [uniprot](https://www.uniprot.org/), [GEO](https://www.ncbi.nlm.nih.gov/geo/).

## Environment
Install python3 for running this code. And these packages should be satisfied:
* tensorflow == 1.14.0
* keras == 2.3.0
* pandas == 1.1.5
* numpy == 1.19.3
* scikit-learn == 0.24.2
* pai4sk == 0.1.5

## Usage
To run the model, default 5 fold cross validation
```
python code/CellComm.py
```
