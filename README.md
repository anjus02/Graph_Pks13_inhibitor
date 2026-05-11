# Pks13 Inhibitor Prediction using GNN

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-red)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green)
![Status](https://img.shields.io/badge/Status-Active-success)

This repository contains a Graph Neural Network (GNN)-based prediction pipeline for identifying potential **Pks13 inhibitors** from molecular SMILES strings. The model predicts whether a compound is a **Pks13 inhibitor** or **non-inhibitor** and generates molecular interpretation results using **GNNExplainer**.

---

## Features

- Predicts Pks13 inhibitors from SMILES
- Graph Neural Network (GNN)-based model
- Molecular graph interpretation using GNNExplainer
- Highlights top 10 important atoms
- Batch prediction from CSV input
- Saves prediction results and molecular images automatically

---

## Requirements

Install the following Python packages before running the prediction script:

```bash
pip install numpy pandas rdkit scikit-learn keras torch torch-geometric pillow
```

---

## Repository Structure

```text
project_folder/
│
├── prediction_script.py
├── gnn_model.pt
├── input.csv
├── feature_normalization.pt
├── Supplementary Files (S1: Dataset, S2: Molecular docking results, S3: Screened FDA drugs)
└── result/
```

---

## Input File

Create an `input.csv` file containing a column named `SMILES`.

Example:

```csv
SMILES
CCO
CCN(CC)CC
```

Save the `input.csv` file in the same folder as `prediction.py`.

---

## Running the Prediction

1. Download or clone the repository.
2. Place `input.csv` in the project directory.
3. Run the prediction script:

```bash
python prediction.py
```

---

## Output

After execution, a folder named `result` will be generated containing:

### `predictions_with_gradcam.csv`

Prediction results including:

- SMILES
- Predicted class
- Prediction probability
- GNNExplainer top 10 important atoms
- Atom importance scores
- Highlighted atom labels

### `gnnexplainer_images/`

Molecular structure images with the top 10 important atoms highlighted using GNNExplainer.

---

## Prediction Classes

| Label | Meaning |
|------|------|
| 0 | Non-Inhibitor |
| 1 | Inhibitor |

---

## Notes

- The input CSV must contain a valid `SMILES` column.
- Invalid SMILES entries will be skipped automatically.
- The trained model file (`gnn_model.pt`) must be present in the project directory before running predictions.

---

## Keywords

`Pks13` `Tuberculosis` `Graph Neural Network` `GNN` `Drug Discovery` `Cheminformatics` `Molecular Property Prediction` `Deep Learning` `RDKit` `PyTorch Geometric` `GNNExplainer`

---

## Publication

If you use this repository or the associated model in your research, please cite the related publication:

```text
Vineet Diwakar, Anju Sharma, Prabha Garg. Graph Neural Network–Driven Virtual Screening of FDA-Approved Drugs Targeting Pks13 in Mycobacterium tuberculosis. Journal Name. Year.
```

DOI:

```text
DOI: 
```

---
