# Multimodal Binding Affinity Prediction

A multi-modal deep learning pipeline for drug-target binding affinity prediction (pKd), combining molecular graph neural networks, protein language model embeddings, and physics-based chemical descriptors.

## Overview

Given a small molecule (SMILES) and a protein sequence, this model predicts their binding affinity as **pKd**.

The pipeline fuses three complementary modalities:

| Modality | Tool | Representation |
|---|---|---|
| Ligand (small molecule) | ChemProp 2.x D-MPNN | Molecular graph |
| Protein | ESM-C 300M (frozen) | 960-dim sequence embedding |
| Physics features | RDKit | 14 physicochemical descriptors |

## Architecture

```
SMILES ──► ChemProp BondMessagePassing ──► mol_proj (256-dim)  ─┐
                                                                  ├─► concat (576-dim) ──► MLP head ──► pKd
Protein seq ──► ESM-C (frozen) ──────────► prot_proj (256-dim) ─┤
                                                                  │
RDKit descriptors ───────────────────────► phys_proj (64-dim)  ──┘
```

Per-modality LayerNorm before concat addresses the magnitude mismatch between modalities.

## Physics-Informed Training Loss

Beyond using RDKit descriptors as input features, physical knowledge is embedded directly into the **training loss** via a pairwise ranking constraint.

Classical empirical scoring functions (ChemScore, AutoDock Vina) decompose ΔG as:

```
ΔG ≈ ΔG_HBond  +  ΔG_hydrophobic  +  ΔG_desolvation  +  ΔG_entropy
```

We approximate this with RDKit descriptors and build a **physics scoring function**:

| Physical term | RDKit proxy | Weight |
|---|---|---|
| H-bond donors/acceptors | `NumHDonors`, `NumHAcceptors` | +0.30 each |
| Hydrophobic burial | `MolLogP` | +0.20 |
| Desolvation penalty | `TPSA` | −0.15 |
| Rotational entropy | `NumRotatableBonds` | −0.20 |
| van der Waals contacts | `MolMR` | +0.15 |

The **pairwise ranking loss** enforces that if the physics score rates molecule *i* much higher than *j*, the model's prediction should also rank *i* above *j*:

```
L_phys = mean  ReLU( pred_j − pred_i )   over pairs (i,j) where score_i − score_j > margin
```

Total loss:

```
L_total = L_MSE  +  λ · L_phys        (λ = 0.1)
```

This makes the model's **ranking** consistent with classical physical chemistry, which is especially valuable for virtual screening where relative ordering matters more than absolute values.

## Key Features

- **Scaffold split** (Bemis-Murcko) for realistic generalization evaluation
- **Deep ensemble** (n=5) for uncertainty estimation
- **Uncertainty calibration** assessed via `uncertainty-toolbox`
- Cosine LR schedule with warmup + early stopping

## Dataset

[TDC](https://tdcommons.ai/) `BindingDB_Kd` (~50K drug-target pairs). Subsampled to 5K for MVP speed.

## Dependencies

```bash
pip install PyTDC chemprop>=2.0 esm>=3.0 rdkit uncertainty-toolbox scikit-learn>=1.4
```

ESM-C requires a HuggingFace token ([EvolutionaryScale/esmc-300m-2024-12](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12)).

## Results (scaffold split, 5K subsample)

The model learns the mid-range trend but shows regression-to-the-mean for extreme binders, which is expected at this data scale. Ensemble uncertainty is underestimated (miscalibration area ≈ 0.37).

## Notes

This project uses physics knowledge at **two levels**:
1. **Physics-informed features** — RDKit descriptors as model input (physicochemical properties)
2. **Physics-informed loss** — pairwise ranking constraint based on a classical scoring function proxy, embedded directly in the training objective

This is distinct from PINN (Physics-Informed Neural Networks), which embed PDE residuals as loss constraints. Here the physics prior is a classical binding energy scoring function rather than a differential equation.
