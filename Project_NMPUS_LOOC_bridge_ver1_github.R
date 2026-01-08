## ================================================================
## LOO LSTM on 7 x T similarity representation (anchors = 7 items)
## ================================================================

## ----------------------------
## 0) Libraries
## ----------------------------
library(reticulate)
library(tidyverse)
library(stringi)
library(pROC)

path_interview <- ""

Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

## ----------------------------
## 1) Reticulate env + imports
## ----------------------------
env <- "r-sentence-transformers"
reticulate::use_condaenv(env, required = TRUE)

sentence_transformers <- import("sentence_transformers", delay_load = TRUE)
torch <- import("torch", delay_load = TRUE)
nn <- import("torch.nn", delay_load = TRUE)

model <- sentence_transformers$SentenceTransformer("paraphrase-MiniLM-L6-v2")
print(model)

## ----------------------------
## 2) Load data + clean + sentence tokenize
## ----------------------------
setwd(path_interview)
data_int <- read.csv("data.csv", header = TRUE, stringsAsFactors = FALSE) %>% as.data.frame()

clean_text <- function(text) {
  text <- as.character(text)
  text <- gsub("\\\\n|\\\\t|\r|\n", " ", text)
  text <- gsub("\\s*\\[(.*?)\\]\\s*", " \\1 ", text)
  text <- gsub("=", "", text)
  text <- gsub("\\\\\"", "", text)
  text <- gsub("[\"\\=]", "", text)
  text <- gsub("c\\(", "", text)
  text <- gsub("\\)", "", text)
  text <- gsub("\\s+", " ", text)
  trimws(text)
}

text_raw   <- stringi::stri_enc_toutf8(data_int$text1, is_unknown_8bit = TRUE)
text_clean <- vapply(text_raw, clean_text, character(1))
text_clean <- tolower(text_clean)

sentence_token <- stringi::stri_split_regex(text_clean, "(?<=[\\.!\\?])\\s+")
sentence_token <- lapply(sentence_token, function(x) {
  x <- trimws(x)
  x[nchar(x) > 0]
})

cat("Sentence counts summary:\n")
print(summary(lengths(sentence_token)))

## ----------------------------
## 3) Define 7 anchor sentences
## ----------------------------
reference_sentences <- c(
  "I read extra material around topics so that I am as knowledgeable as possible.",
  "I use time wisely to achieve my academic goals.",
  "I use peer and lecturer information to improve my work.",
  "I attempt to master most of the material I learn at school.",
  "I will do my best in my field of study.",
  "I have my own unique strategy to achieve my academic goals.",
  "Ability is the key to academic success."
)

## Encode anchors (384-dim each)
ref_embed_mat <- do.call(rbind, lapply(reference_sentences, function(s) model$encode(s)))

## ----------------------------
## 4) Build 7 x T matrices (cosine similarity to anchors)
## ----------------------------
cosine_similarity_vec <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

# For each text: matrix [T_i, 7]
sim_mats <- vector("list", length(sentence_token))

for (ti in seq_along(sentence_token)) {
  sents <- sentence_token[[ti]]
  if (length(sents) == 0) {
    sim_mats[[ti]] <- matrix(0, nrow = 1, ncol = length(reference_sentences))
    next
  }
  
  sent_embeds <- lapply(sents, function(s) model$encode(s))
  
  M <- matrix(NA_real_, nrow = length(sent_embeds), ncol = length(reference_sentences))
  for (i in seq_along(sent_embeds)) {
    for (j in seq_len(nrow(ref_embed_mat))) {
      M[i, j] <- cosine_similarity_vec(sent_embeds[[i]], ref_embed_mat[j, ])
    }
  }
  sim_mats[[ti]] <- M
}

## ----------------------------
## 5) Pad to common T_max and build tensor [n, T_max, 7]
## ----------------------------
pad_mats <- function(mats, pad_value = 0) {
  T_max <- max(vapply(mats, nrow, integer(1)))
  lapply(mats, function(M) {
    out <- matrix(pad_value, nrow = T_max, ncol = ncol(M))
    out[seq_len(nrow(M)), ] <- M
    out
  })
}

sim_mats_pad <- pad_mats(sim_mats, pad_value = 0)

input_array <- array(
  unlist(sim_mats_pad),
  dim = c(length(sim_mats_pad), nrow(sim_mats_pad[[1]]), ncol(sim_mats_pad[[1]]))
)

x_tensor <- torch$tensor(input_array, dtype = torch$float32)
cat("x_tensor shape:\n")
print(x_tensor$shape)  # [n, T_max, 7]

## ----------------------------
## 6) Labels (map to 0..K-1)
## ----------------------------
label_col <- "rated"   # <<< change to "reported" etc if needed
y_raw <- as.integer(data_int[[label_col]])

lvl <- sort(unique(y_raw))
map <- setNames(seq_along(lvl) - 1L, lvl)
y_vec <- unname(map[as.character(y_raw)])
num_classes <- length(lvl)

y_tensor <- torch$tensor(y_vec)$to(dtype = torch$int64)

cat("Classes (original -> remapped):\n")
print(map)

## ----------------------------
## 7) Python model (LSTM classifier)
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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)          # [B, T, H]
        last_hidden = lstm_out[:, -1, :]    # [B, H]
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
## 8) Metrics
## ----------------------------
macro_f1_simple <- function(truth, pred, K) {
  f1s <- numeric(K)
  for (k in 0:(K-1)) {
    tp <- sum(pred == k & truth == k)
    fp <- sum(pred == k & truth != k)
    fn <- sum(pred != k & truth == k)
    prec <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    rec  <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1s[k + 1] <- if ((prec + rec) == 0) 0 else 2 * prec * rec / (prec + rec)
  }
  mean(f1s)
}

macro_ovr_auc <- function(truth, prob_mat) {
  K <- ncol(prob_mat)
  aucs <- rep(NA_real_, K)
  for (k in 1:K) {
    yk <- as.integer(truth == (k - 1L))
    if (length(unique(yk)) < 2) {
      aucs[k] <- NA_real_
    } else {
      aucs[k] <- as.numeric(pROC::roc(yk, prob_mat[, k], quiet = TRUE)$auc)
    }
  }
  mean(aucs, na.rm = TRUE)
}

## ----------------------------
## 9) One LOO run (returns acc/f1/auc + preds/probs)
## ----------------------------
run_loo_once <- function(seed = 1234, epochs = 300, lr = 0.005, hidden_dim = 2, dropout = 0.2) {
  py$reset_seed(as.integer(seed))
  
  shp <- as.integer(reticulate::iterate(x_tensor$size()))
  n <- shp[1]
  T_max <- shp[2]
  D <- shp[3]  # should be 7
  
  truth <- as.integer(y_vec)
  
  preds <- integer(n)
  probs_all <- matrix(NA_real_, nrow = n, ncol = num_classes)
  
  for (i in 1:n) {
    test_idx0 <- as.integer(i - 1L)
    all_idx0  <- 0:(n - 1L)
    train_idx0 <- as.integer(all_idx0[all_idx0 != test_idx0])
    
    idx_train <- torch$tensor(train_idx0, dtype = torch$long)
    idx_test  <- torch$tensor(c(test_idx0), dtype = torch$long)
    
    x_train <- torch$index_select(x_tensor, dim = 0L, index = idx_train)
    y_train <- torch$index_select(y_tensor, dim = 0L, index = idx_train)
    
    model <- TextClassifier(py_int(D), py_int(hidden_dim), py_int(num_classes), dropout = dropout)
    optimizer <- torch$optim$Adam(model$parameters(), lr = lr)
    criterion <- nn$CrossEntropyLoss()
    
    model$train()
    for (e in 1:epochs) {
      out <- model(x_train)
      logits <- out[[1]]
      loss <- criterion(logits, y_train)
      optimizer$zero_grad(); loss$backward(); optimizer$step()
    }
    
    x_test <- torch$index_select(x_tensor, dim = 0L, index = idx_test)
    
    model$eval()
    with(torch$no_grad(), {
      out_test <- model(x_test)
      logits <- out_test[[1]]
      prob <- torch$softmax(logits, dim = 1L)
      pred <- torch$argmax(logits, dim = 1L)
    })
    
    preds[i] <- as.integer(pred$cpu()$numpy())
    probs_all[i, ] <- as.numeric(prob$cpu()$numpy())
  }
  
  acc <- mean(preds == truth)
  f1  <- macro_f1_simple(truth, preds, num_classes)
  
  if (num_classes == 2) {
    # class "1" prob = column 2
    auc <- if (length(unique(truth)) < 2) NA_real_ else as.numeric(pROC::roc(truth, probs_all[,2], quiet=TRUE)$auc)
  } else {
    auc <- macro_ovr_auc(truth, probs_all)
  }
  
  list(acc = acc, f1 = f1, auc = auc, preds = preds, probs = probs_all, truth = truth)
}

## ----------------------------
## 10) Repeat LOO across seeds (optional stability)
## ----------------------------
trials <- 100     # set to 1 if you want only one run
epochs <- 300
lr <- 0.005
hidden_dim <- 2

acc_vec <- numeric(trials)
f1_vec  <- numeric(trials)
auc_vec <- numeric(trials)

for (t in 1:trials) {
  cat("\n=== LOO Trial", t, "===\n")
  res <- run_loo_once(seed = 1000 + t, epochs = epochs, lr = lr, hidden_dim = hidden_dim)
  acc_vec[t] <- res$acc
  f1_vec[t]  <- res$f1
  auc_vec[t] <- res$auc
  cat(sprintf("Acc: %.4f | Macro-F1: %.4f | AUC: %.4f\n", res$acc, res$f1, res$auc))
}

cat("\nSUMMARY (mean ± SD):\n")
cat(sprintf("Accuracy: %.3f ± %.3f\n", mean(acc_vec, na.rm=TRUE), sd(acc_vec, na.rm=TRUE)))
cat(sprintf("Macro-F1 : %.3f ± %.3f\n", mean(f1_vec,  na.rm=TRUE), sd(f1_vec,  na.rm=TRUE)))
cat(sprintf("AUC      : %.3f ± %.3f\n", mean(auc_vec, na.rm=TRUE), sd(auc_vec, na.rm=TRUE)))

## ----------------------------
## 11) Save core objects
## ----------------------------
save(
  sentence_token,
  reference_sentences,
  ref_embed_mat,
  sim_mats, sim_mats_pad,
  x_tensor,
  y_vec, num_classes, map,
  acc_vec, f1_vec, auc_vec,
  file = file.path(path_interview, "loo_7xT_results.RData")
)

cat("\nSaved:", file.path(path_interview, "loo_7xT_results.RData"), "\n")