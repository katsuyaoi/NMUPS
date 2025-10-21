This repository reproduces the analyses for “Bridge Embeddings for Mixed-Methods Analysis: Repurposing S-BERT to Link Control Narratives and Stimulant Use.” It includes:
•	An LSTM classifier using bridge embeddings (theory-guided, 7-D)
•	The same LSTM using off-the-shelf S-BERT (384-D)
•	Post-estimation scripts to generate tables/figures reported in the manuscript
Data
Place the following files in the same directory as the scripts:
•	data.csv — interview/narrative data
•	surveydata.csv — survey data for the baseline quantitative analysis
The scripts assume these filenames and relative paths. If yours differ, edit the path variables at the top of each script.
Files (scripts)
•	Supp-codes_LSTM_bridge.R — LSTM with bridge embeddings
•	Supp-codes_LSTM_BERT.R — LSTM with untuned S-BERT embeddings
•	Supp_codes_LSTM_post-estimate.R — post-estimation (metrics summaries, figures, tables)
Software requirements
R (≥ 4.2) with packages:
reticulate, readr, dplyr, tidyr, purrr, tibble, stringr, ggplot2, yardstick, pROC
Python (via reticulate):
python 3.10, torch==2.0.1, transformers>=4.29.0, sentence-transformers>=2.3.0, numpy, scikit-learn

