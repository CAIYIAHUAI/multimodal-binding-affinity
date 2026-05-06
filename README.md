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

This project uses **physics-informed features** (RDKit descriptors as model input) rather than PINN-style physics constraints embedded in the loss. The term "physics-informed" here refers to the feature augmentation strategy.
