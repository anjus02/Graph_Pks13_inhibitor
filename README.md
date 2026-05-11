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
├── prediction.py
├── training_advanced.py
├── Smiles_to_graph.py
├── gnn_model.pt
├── input.csv
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
