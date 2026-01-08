## ================================================================
## LOO LSTM on RAW sentence embeddings (no cosine; 384-dim per sentence)
## Each text -> [T_i, 384] then pad -> [n, T_max, 384]
## LOO evaluation with ACC / Macro-F1 / AUC across trials
## ================================================================

## ----------------------------
## 0) Libraries
## ----------------------------
library(reticulate)
library(tidyverse)
library(stringi)
library(pROC)

path_interview <- ""
setwd(path_interview)
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

## ----------------------------
## 1) Reticulate env + imports
## ----------------------------
reticulate::use_condaenv("r-sentence-transformers", required = TRUE)

sentence_transformers <- reticulate::import("sentence_transformers", delay_load = TRUE)
torch <- reticulate::import("torch", delay_load = TRUE)
nn <- reticulate::import("torch.nn", delay_load = TRUE)

model <- sentence_transformers$SentenceTransformer("paraphrase-MiniLM-L6-v2")
print(model)

## ----------------------------
## 2) Load data + clean text + sentence tokenize
## ----------------------------
data <- read.csv("data.csv", header = TRUE, stringsAsFactors = FALSE) %>% as.data.frame()

stopifnot("text1" %in% names(data))   # change if needed

clean_text <- function(x) {
  x <- as.character(x)
  x <- stringi::stri_enc_toutf8(x, is_unknown_8bit = TRUE)
  x <- gsub("\\\\n|\\\\t|\r|\n", " ", x)
  x <- gsub("\\s*\\[(.*?)\\]\\s*", " \\1 ", x)
  x <- gsub("=", "", x)
  x <- gsub("\\\\\"", "", x)
  x <- gsub("[\"\\=]", "", x)
  x <- gsub("c\\(", "", x)
  x <- gsub("\\)", "", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

text_clean <- vapply(data$text1, clean_text, character(1))
text_clean <- tolower(text_clean)

# sentence split (keeps . ! ?)
sentence_token <- stringi::stri_split_regex(
  text_clean,
  "(?<=[\\.!\\?])\\s+"
)

sentence_token <- lapply(sentence_token, function(s) {
  s <- trimws(s)
  s[nchar(s) > 0]
})

cat("Sentence counts summary:\n")
print(summary(lengths(sentence_token)))

## ----------------------------
## 3) Encode sentences -> list of matrices [T_i, 384]
## ----------------------------
# If you want speed, you can batch-encode per text;
# keeping per-sentence for clarity.
embed_list <- vector("list", length(sentence_token))

for (i in seq_along(sentence_token)) {
  sents <- sentence_token[[i]]
  if (length(sents) == 0) {
    # keep at least 1 row to avoid shape issues
    embed_list[[i]] <- matrix(0, nrow = 1, ncol = 384)
    next
  }
  E <- do.call(rbind, lapply(sents, function(s) model$encode(s)))
  embed_list[[i]] <- E
}

T_max <- max(vapply(embed_list, nrow, integer(1)))
D     <- ncol(embed_list[[1]])

cat("T_max =", T_max, " | D =", D, "\n")

# lengths vector (true sentence counts)
lengths_vec <- vapply(embed_list, nrow, integer(1))

# pad to [T_max, D]
pad_one <- function(M, T_max) {
  out <- matrix(0, nrow = T_max, ncol = ncol(M))
  out[seq_len(nrow(M)), ] <- M
  out
}

padded <- lapply(embed_list, pad_one, T_max = T_max)

# tensor [n, T_max, D]
input_array <- array(unlist(padded), dim = c(length(padded), T_max, D))
input_tensor <- torch$tensor(input_array, dtype = torch$float32)

cat("input_tensor shape:\n")
print(input_tensor$shape)

## ----------------------------
## 4) Labels (map to 0..K-1)
## ----------------------------
label_col <- "reported"   # <<< change if needed
stopifnot(label_col %in% names(data))

y_raw <- as.integer(as.character(data[[label_col]]))

lvl <- sort(unique(y_raw))
map <- setNames(seq_along(lvl) - 1L, lvl)
y_vec <- unname(map[as.character(y_raw)])
num_classes <- length(lvl)

y_tensor_full <- torch$tensor(y_vec)$to(dtype = torch$int64)

cat("Class map (original -> 0..K-1):\n")
print(map)

## ----------------------------
## 5) Python: LSTM classifier + seed reset
## IMPORTANT: use last REAL timestep (not padded last row)
## ----------------------------
reticulate::py_run_string("
import torch
import torch.nn as nn
import numpy as np, random

class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout=0.2):
        super(TextClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # x: [B, T, D], lengths: [B] (CPU tensor ok)
        lstm_out, _ = self.lstm(x)  # [B, T, H]

        # gather last valid timestep for each row: index = lengths-1
        idx = (lengths - 1).clamp(min=0)
        idx = idx.view(-1, 1, 1).expand(lstm_out.size(0), 1, lstm_out.size(2))
        last_hidden = lstm_out.gather(1, idx).squeeze(1)  # [B, H]

        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits, last_hidden

def reset_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
")

TextClassifier <- py$TextClassifier
py_int <- function(x) reticulate::py_eval(sprintf("int(%d)", x))

## ----------------------------
## 6) Metrics
## ----------------------------
macro_f1_fn <- function(true, pred, k) {
  f1s <- numeric(k)
  for (c in 0:(k-1)) {
    tp <- sum(pred == c & true == c)
    fp <- sum(pred == c & true != c)
    fn <- sum(pred != c & true == c)
    prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1s[c + 1] <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
  }
  mean(f1s)
}

compute_auc <- function(y_true, y_prob) {
  k <- ncol(y_prob)
  if (k == 2) {
    roc_obj <- pROC::roc(
      response  = factor(y_true, levels = c(0, 1)),
      predictor = y_prob[, 2],
      quiet = TRUE
    )
    as.numeric(pROC::auc(roc_obj))
  } else {
    lev_chr <- as.character(0:(k - 1))
    y_prob_df <- as.data.frame(y_prob)
    colnames(y_prob_df) <- lev_chr
    mc <- pROC::multiclass.roc(
      response  = factor(y_true, levels = 0:(k - 1)),
      predictor = y_prob_df,
      quiet = TRUE
    )
    as.numeric(mc$auc)
  }
}

## ----------------------------
## 7) One LOO run
## ----------------------------
run_loo_once <- function(seed = 1234, epochs = 300, lr = 0.005, hidden_dim = 2, dropout = 0.2) {
  py$reset_seed(as.integer(seed))
  
  n <- as.integer(input_tensor$shape[0])
  D <- as.integer(input_tensor$shape[2])
  
  truth <- as.integer(y_vec)
  
  results_pred <- integer(n)
  results_prob <- matrix(NA_real_, nrow = n, ncol = num_classes)
  
  # lengths tensor (CPU long)
  lengths_t_all <- torch$tensor(as.integer(lengths_vec), dtype = torch$long)
  
  for (i in 1:n) {
    test_idx0  <- as.integer(i - 1L)
    all_idx0   <- 0:(n - 1L)
    train_idx0 <- as.integer(all_idx0[all_idx0 != test_idx0])
    
    idx_train_t <- torch$tensor(train_idx0, dtype = torch$long)
    idx_test_t  <- torch$tensor(c(test_idx0), dtype = torch$long)
    
    x_train <- torch$index_select(input_tensor, dim = 0L, index = idx_train_t)
    y_train <- torch$index_select(y_tensor_full, dim = 0L, index = idx_train_t)
    len_train <- torch$index_select(lengths_t_all, dim = 0L, index = idx_train_t)
    
    model <- TextClassifier(py_int(D), py_int(hidden_dim), py_int(num_classes), dropout = dropout)
    optimizer <- torch$optim$Adam(model$parameters(), lr = lr)
    criterion <- nn$CrossEntropyLoss()
    
    model$train()
    for (e in 1:epochs) {
      out <- model(x_train, len_train)
      logits <- out[[1]]
      loss <- criterion(logits, y_train)
      optimizer$zero_grad(); loss$backward(); optimizer$step()
    }
    
    x_test <- torch$index_select(input_tensor, dim = 0L, index = idx_test_t)
    len_test <- torch$index_select(lengths_t_all, dim = 0L, index = idx_test_t)
    
    model$eval()
    with(torch$no_grad(), {
      out_test <- model(x_test, len_test)
      logits <- out_test[[1]]
      probs  <- torch$softmax(logits, dim = 1L)
      pred   <- torch$argmax(logits, dim = 1L)
    })
    
    results_pred[i] <- as.integer(pred$cpu()$numpy())
    results_prob[i, ] <- as.numeric(probs$cpu()$numpy())
  }
  
  acc <- mean(results_pred == truth)
  f1  <- macro_f1_fn(truth, results_pred, num_classes)
  auc <- compute_auc(truth, results_prob)
  
  list(acc = acc, f1 = f1, auc = auc,
       preds = results_pred, probs = results_prob, truth = truth)
}

## ----------------------------
## 8) Multiple trials (stability over seeds)
## ----------------------------
trials <- 100
epochs_per_trial <- 300
lr <- 0.005
hidden_dim <- 2

acc_vec <- numeric(trials)
f1_vec  <- numeric(trials)
auc_vec <- numeric(trials)

for (t in 1:trials) {
  cat("\n=== Trial", t, "===\n")
  res <- run_loo_once(seed = 2000 + t, epochs = epochs_per_trial, lr = lr, hidden_dim = hidden_dim)
  acc_vec[t] <- res$acc
  f1_vec[t]  <- res$f1
  auc_vec[t] <- res$auc
  cat(sprintf("Trial %d — Acc: %.4f | F1: %.4f | AUC: %.4f\n",
              t, acc_vec[t], f1_vec[t], auc_vec[t]))
}

cat(sprintf("\n[RAW-384] Accuracy (mean ± SD): %.3f ± %.3f\n", mean(acc_vec), sd(acc_vec)))
cat(sprintf("[RAW-384] Macro-F1 (mean ± SD): %.3f ± %.3f\n", mean(f1_vec),  sd(f1_vec)))
cat(sprintf("[RAW-384] AUC (mean ± SD): %.3f ± %.3f\n", mean(auc_vec), sd(auc_vec)))

save(
  sentence_token,
  lengths_vec,
  input_tensor,
  y_vec, num_classes, map,
  acc_vec, f1_vec, auc_vec,
  file = file.path(path_interview, "results_raw384_LOO_100.RData")
)

cat("\nSaved:", file.path(path_interview, "results_raw384_LOO_100.RData"), "\n")
