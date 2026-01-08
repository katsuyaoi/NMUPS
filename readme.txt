This lists embedding/classification tasks of interview data via LSTM
How to use:
downlaod Data
Place the following files in the same directory as the scripts:
•	data.csv — interview/narrative data
•	surveydata.csv — survey data for the baseline quantitative analysis
The scripts assume these filenames and relative paths. If yours differ, edit the path variables at the top of each script: path_interview
Files (scripts)
•	Project_NMPUS_LOOC_bridge_ver1_github.R — LSTM with bridge S-BERT embeddings
•	Project_NMPUS_LOOC_naive_ver1_github.R — LSTM with untuned S-BERT embeddings

Software requirements
R (≥ 4.2) with packages:
reticulate, readr, dplyr, tidyr, purrr, tibble, stringr, ggplot2, yardstick, pROC
Python (via reticulate):
python 3.10, torch==2.0.1, transformers>=4.29.0, sentence-transformers>=2.3.0, numpy, scikit-learn

Hyperparameters and modeling choices (S-BERT vs. Bridge LSTM)

Both LSTM classifiers were implemented using identical training, architectural, and evaluation settings, with the only substantive difference being the dimensionality and theoretical grounding of the input representations. This design ensures that any observed performance differences reflect representational constraints rather than hyperparameter tuning.

Representation
Sentence encoder: paraphrase-MiniLM-L6-v2 (Sentence-BERT) for both models
Unit of analysis: Sentence-level tokenization for both models
Per-sentence feature dimensionality:
S-BERT (baseline): 384-dimensional raw semantic embeddings
Bridge (theory-guided): 7-dimensional sentence-by-item similarity vectors aligned with survey anchors
Padding value: 0 for all sequences (identical padding strategy)
The key experimental contrast is raw semantic space versus theory-anchored representational space, not differences in model capacity or training regime.

Architecture
Model type: Single-layer LSTM (PyTorch default)
Number of LSTM layers: 1
Bidirectionality: Disabled (unidirectional LSTM)
Hidden state size (H): 2 units for both models
Dropout probability: 0.2 applied to the final hidden state
Output head: Linear projection from hidden state (H) to class logits (K)
The intentionally small hidden dimension fixes model capacity, allowing representational structure—rather than expressive depth—to drive differences in classification consistency.

Training
Optimizer: Adam
Learning rate: 0.005
Loss function: Cross-entropy
Training budget: 300 epochs per fold
All training settings were held constant across models.

Evaluation
Validation scheme: Leave-One-Out cross-validation (LOO-CV)
Number of trials: 100 random initializations (or 1,000 in extended stability analyses)
Evaluation metrics: Accuracy, Macro-F1, Area Under the ROC Curve (AUC)
LOO-CV was selected as a necessity for small, imbalanced samples, ensuring that every observation serves as a held-out test case while preserving a strict train/test separation.



