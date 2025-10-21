### objective of this code
### extract survey items based on an existing questinnaire, 
### convert general sentence transformer to get sentence embedding
### calculate item similarity based on embedding
### update the embedding, based on target item similiarty (possibly using item correlation)
### note that beween-item correlation is just an example. need be edited appropriatedly 



##### current architecture  
#####






install.packages("tm")           # For text mining and preprocessing
install.packages("text2vec")     # For text vectorization
install.packages("tidyverse")    # For data manipulation
install.packages("stringi")
install.packages("tokenizers")
install.packages("irlba")
install.packages("reticulate")
install.packages("textclean")
install.packages("dtw")
###reticulate::install_miniconda()
library("dtw")
library(reticulate)
library(tm)           # For text mining and preprocessing
library(text2vec)     # For text vectorization
library(tidyverse)    # For data manipulation
library(stringi)
library(tokenizers)
library(irlba)
library(textclean)





reticulate::use_condaenv("r-sentence-transformers", required = TRUE)

sentence_transformers <- reticulate::import("sentence_transformers")
torch <- reticulate::import("torch")
nn <- reticulate::import("torch.nn")

# Load MiniLM (BERT-family transformer)
model <- sentence_transformers$SentenceTransformer('paraphrase-MiniLM-L6-v2')

### base architechture of  the model
## 1) break down text into sentence levevls 
## convert setence levels into latent construct values 
## each dimention represents similartiy to each sentence in the questionnaire
## now each text has a matrix
## Use BERT to condense, make sure that hidden layers are able to project matrices back close enough to the original (training)
## use layers as a text representation 
## get distance between texts using trained layers 


# Use pre-trained BERT for text classification



############# to be implemented 
############ not just use sentence items from pschometrics but also responsese 
########### so you have the person's representation as a list of scalars attached to a sentence 



###################3

setwd("datapath")
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

### here is template code for any csv files that store text separately in each row. 
data <- read.csv("data.csv", header = TRUE)
data <- data.frame(data)



# Tokenize the 'text1' column by sentence
library(stringi)
text <- data$text1
tokenized_text <- stri_split_boundaries(text, type = "sentence")
#text <- tokenize_words(text)
# Separate features (sentences) and labels (sentiment scores)
# defining X (text) and Y (number)


### iterate this clean function, run it all the way till vocab$term to see until you eliminate undesirables
clean_text <- function(text) {
  # 1. Remove numbers 
  text <- gsub("=", "", text)
  # 3. Remove all non-alphabetic characters, but KEEP periods
#  text <- gsub("[^a-zA-Z\\.\\s]", " " , text)  # Retain periods
  
  # 2. Remove newlines and other escape sequences
  text <- gsub("\\\\n", " ", text)   # Remove newline escape sequences
  text <- gsub("\\\\t", " ", text)   # Remove tab escape sequences
  text <- gsub("\n", "", text)      # Replace actual newline characters with spaces
  text <- gsub("\n\"", "", text)    # Remove escaped newlines
  text <- gsub("\\\"", "", text)     # Remove escaped quotes
  

  # 4. Additional cleaning: remove extra quotes, equal signs, and trim whitespace
  text <- gsub("[\"\\=]", "", text)          # Remove remaining undesirables
  text <- gsub("c\\(|\\(|\\)|\\[|\\]", "", text)  # Remove "c(", parentheses, and square brackets
  
  # 2. Remove multiple spaces
  text <- gsub("\\s+", " ", text)  # Replace multiple spaces with a single space
  
  # 3. Remove spaces before periods
  text <- gsub("\\s+\\.", ".", text) 
  text <- gsub("\\s+\\.", ".", text) 
  text <- gsub("\\s+([\\.,])", "\\1", text)
  text <- gsub("^\\s*,", "", text)  # Remove commas at the start of the text
  text <- gsub("\\.\\s*,", ".", text)
  text <- gsub(",,", "", text)
  text <- trimws(text)  # Trim leading/trailing spaces
  return(text)
}


corpus <- Corpus(VectorSource(tokenized_text))
corpus <- tm_map(corpus, content_transformer(tolower))
# corpus <- tm_map(corpus, removeWords, stopwords("english"))

corpus$content

corpus <- tm_map(corpus, content_transformer(clean_text))
corpus$content

inspect(corpus) 
doc_lengths <- sapply(corpus, function(doc) nchar(as.character(doc)))
print(doc_lengths)

# Calculate document embeddings by averaging word embeddings
sentence_token <- stri_split_regex(corpus$content, "\\.\\s*")




# --- Encode sentences into BERT embeddings ---
embedding_lists <- lapply(sentence_token, function(sentences) {
  lapply(sentences, function(s) model$encode(s))
})

# --- Pad embeddings so all texts have same length ---
max_len <- max(sapply(embedding_lists, length))
embedding_dim <- length(embedding_lists[[1]][[1]])

pad_embeddings <- function(embed_list, max_len, embedding_dim) {
  n <- length(embed_list)
  mat <- matrix(0, nrow = max_len, ncol = embedding_dim)
  for (i in 1:n) {
    mat[i, ] <- embed_list[[i]]
  }
  return(mat)
}

padded_embeddings <- lapply(embedding_lists, pad_embeddings, max_len = max_len, embedding_dim = embedding_dim)

# --- Convert to torch tensor [batch, seq_len, embedding_dim] ---
input_array <- array(unlist(padded_embeddings), dim = c(length(padded_embeddings), max_len, embedding_dim))
input_tensor <- torch$tensor(input_array, dtype = torch$float32)

print(input_tensor$shape)




install.packages("plotly")
install.packages("zoo")



###################### re-ryn here if necessary

### just re-categorizing 1 to 2 because the key distinction is whether they consider the need to use it. 

#### be sure to run this 
setwd("datapath")
Sys.setlocale("LC_CTYPE", "en_US.UTF-8")

### here is template code for any csv files that store text separately in each row. 
data <- read.csv("data.csv", header = TRUE)
data <- data.frame(data)


data$rated[data$rated ==1] <- 2 
data$"rated" <- as.integer(data$"rated") - 2
data$"reported" <- as.integer(data$"reported") - 1
data$"assigned" <- as.integer(data$"assigned") - 1



############ we just want to condense lstm_output into desirable shapes 
# Pass the input through the LSTM model


# Define the classifier (same as before)

reticulate::py_run_string("
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.classifier(last_hidden)  # logits
        return output, last_hidden
")
TextClassifier <- py$TextClassifier
py_int <- function(x) reticulate::py_eval(sprintf("int(%d)", x))

# -------------------------
# 2) Dimensions / labels (from S-BERT tensor: input_tensor)
# -------------------------
n             <- input_tensor$shape[0]
max_sentences <- input_tensor$shape[1]
embedding_dim <- input_tensor$shape[2]  # e.g., 384 for MiniLM

hidden_dim  <- 2
# Raw labels
y_vector_raw <- as.integer(as.character(data$reported))
# Map to 0..K-1 for Torch consistency
lvl <- sort(unique(y_vector_raw))
map <- setNames(seq_along(lvl) - 1L, lvl)
y_vector <- unname(map[as.character(y_vector_raw)])
num_classes <- length(lvl)

y_tensor_full <- torch$tensor(y_vector)$to(dtype = torch$int64)

# -------------------------
# 3) Metric helpers (macro-F1 and AUC)
# -------------------------
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
  if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")
  library(pROC)
  k <- ncol(y_prob)
  if (k == 2) {
    roc_obj <- pROC::roc(
      response  = factor(y_true, levels = c(0, 1)),
      predictor = y_prob[, 2],  # prob of class 1
      quiet = TRUE
    )
    as.numeric(pROC::auc(roc_obj))
  } else {
    # name columns "0","1",... to match response levels
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

# -------------------------
# 4) One LOO run; returns metrics + raw preds/probs/true
# -------------------------
run_loo_once <- function(seed = NULL, epochs = 300, lr = 0.005, hidden_dim = 2) {
  if (!is.null(seed)) torch$manual_seed(as.integer(seed))
  
  results_pred <- integer(n)
  results_true <- y_vector
  results_prob <- matrix(NA_real_, nrow = n, ncol = num_classes)
  
  for (i in 1:n) {
    # indices (0-based for Torch)
    test_index0  <- as.integer(i - 1L)
    all_idx0     <- 0:(n - 1L)
    train_idx0   <- as.integer(all_idx0[all_idx0 != test_index0])
    
    # torch index tensors
    idx_train_t <- torch$tensor(train_idx0, dtype = torch$long)
    idx_test_t  <- torch$tensor(c(test_index0), dtype = torch$long)
    
    # slice data from S-BERT tensor
    input_train <- torch$index_select(input_tensor, dim = 0L, index = idx_train_t)
    y_train     <- torch$index_select(y_tensor_full, dim = 0L, index = idx_train_t)
    
    # model
    model <- TextClassifier(
      py_int(embedding_dim),
      py_int(hidden_dim),
      py_int(num_classes)
    )
    optimizer <- torch$optim$Adam(model$parameters(), lr = lr)
    criterion <- nn$CrossEntropyLoss()
    model$train()
    
    for (e in 1:epochs) {
      out    <- model(input_train)
      logits <- out[[1]]
      loss   <- criterion(logits, y_train)
      optimizer$zero_grad(); loss$backward(); optimizer$step()
    }
    
    # test tensor (1 x T x D)
    test_tensor <- torch$index_select(input_tensor, dim = 0L, index = idx_test_t)
    
    # inference
    model$eval()
    with(torch$no_grad(), {
      out_test <- model(test_tensor)
      logits   <- out_test[[1]]
      probs    <- torch$softmax(logits, dim = 1L)
      pred_cls <- torch$argmax(logits, dim = 1L)
    })
    
    results_pred[i]   <- as.integer(pred_cls$cpu()$numpy())
    results_prob[i, ] <- as.numeric(probs$cpu()$numpy())
  }
  
  acc_val <- mean(results_pred == results_true)
  f1_val  <- macro_f1_fn(results_true, results_pred, num_classes)
  auc_val <- compute_auc(results_true, results_prob)
  
  list(
    acc = acc_val,
    f1  = f1_val,
    auc = auc_val,
    preds = results_pred,
    probs = results_prob,
    true  = results_true
  )
}

# -------------------------
# 5) Multiple trials; collect mean ± SD
# -------------------------
trials <- 100
epochs_per_trial <- 300
lr <- 0.005

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

# -------------------------
# 6) Summary for reporting
# -------------------------
acc_mean <- mean(acc_vec); acc_sd <- sd(acc_vec)
f1_mean  <- mean(f1_vec);  f1_sd  <- sd(f1_vec)
auc_mean <- mean(auc_vec); auc_sd <- sd(auc_vec)

cat(sprintf("\n[S-BERT] Accuracy (mean ± SD): %.3f ± %.3f\n", acc_mean, acc_sd))
cat(sprintf("[S-BERT] Macro-F1 (mean ± SD): %.3f ± %.3f\n", f1_mean,  f1_sd))
cat(sprintf("[S-BERT] AUC (mean ± SD): %.3f ± %.3f\n",  auc_mean, auc_sd))

