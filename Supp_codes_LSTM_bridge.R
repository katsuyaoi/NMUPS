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






# (2) Create a clean environment
reticulate::conda_create(envname = "r-sentence-transformers", packages = "python=3.8")

# (3) Install pip inside the environment
reticulate::conda_install(envname = "r-sentence-transformers", packages = "pip")

# (4) Install necessary Python packages with exact working versions
reticulate::conda_install(
  envname = "r-sentence-transformers",
  pip = TRUE,
  packages = c(
    "torch==2.0.1",
    "sentence-transformers>=2.3.0",
    "transformers>=4.29.0",
    "huggingface_hub>=0.14.0"
  )
)

# (5) Restart your R session here! (Ctrl + Shift + F10 in RStudio)

# (6) Use the environment
reticulate::use_condaenv("r-sentence-transformers", required = TRUE)

# (7) Import packages
sentence_transformers <- reticulate::import("sentence_transformers")
torch <- reticulate::import("torch")
nn <- reticulate::import("torch.nn")

# (8) Load the model
model <- sentence_transformers$SentenceTransformer('paraphrase-MiniLM-L6-v2')

# (9) (Optional) Confirm it works
print(model)



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







#################### Define two sentences and make sure you have sentence_token from data
reference_sentences <- c(
  "I read extra material around topics so that I am as knowledgeable as possible.",
  "I use time wisely to achieve my academic goals.",
  "I use peer and lecturer information to improve my work.",
  "I attempt to master most of the material I learn at school.",
  "I will do my best in my field of study.",
  "I have my own unique strategy to achieve my academic goals.",
  "Ability is the key to academic success."
)

####### extracting original reference_embedding

reference_embeddings <- lapply(reference_sentences, function(sentence) {
  model$encode(sentence)  # Generate embedding for each sentence
})


cosine_similarity <- function(a, b) {
  sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))
}

embedding_matrix <- do.call(rbind, reference_embeddings)

num_sentences <- length(reference_embeddings)
similarity_matrix <- matrix(0, nrow = num_sentences, ncol = num_sentences)

for (i in 1:num_sentences) {
  for (j in 1:num_sentences) {
    similarity_matrix[i, j] <- cosine_similarity(embedding_matrix[i, ], embedding_matrix[j, ])
  }
}

# Convert matrix to a data frame for better visualization
similarity_df <- as.data.frame(similarity_matrix)
rownames(similarity_df) <- reference_sentences
colnames(similarity_df) <- reference_sentences

# Print similarity matrix
print(similarity_df)

###### quality check- can large-corpus based sentence translate it so that 
##### similarity between items can be detected ? 


### torch <- reticulate::import("torch")
### nn <- torch$nn
### F <- torch$nn$functional

# Convert the 3D array into a PyTorch tensor



#########################3


# Define target similarity matrix (example)
# it can be based on subpopulation data
### using the matrix from the survey data
setwd("datapath")
library(dplyr)
data <- read.csv("surveydata.csv",  header = TRUE)
cor_matrix <- cor(data[,12:18], use = "pairwise.complete.obs")

target_matrix <- 
  matrix(
    cor_matrix,
    nrow = 7,
    byrow = TRUE
  )

target_similarity <- torch$tensor(matrix(
  cor_matrix,
  nrow = 7, byrow = TRUE
), requires_grad = FALSE) 


# Convert sentence embeddings to torch tensors
embedding_tensor <- torch$tensor(do.call(rbind, reference_embeddings), requires_grad = TRUE)




# Cosine similarity function
cosine_similarity_matrix <- function(embeddings) {
  norm_emb <- embeddings / embeddings$norm(p = 2, dim = 1L, keepdim = TRUE)  
  return(norm_emb$mm(norm_emb$t()))  
}

# Define loss function (Mean Squared Error)
loss_fn <- function(pred_sim, target_sim) {
  loss_value <- ((pred_sim - target_sim)^2)$mean()  # Ensure loss remains a tensor
  return(loss_value)
} 

# Optimizer (RMSprop)
learning_rate <- 0.001
optimizer <- torch$optim$RMSprop(list(embedding_tensor), lr = learning_rate) 

# Training loop
epochs <- 500
for (epoch in 1:epochs) {
  optimizer$zero_grad()
  
  # Compute current similarity
  predicted_similarity <- cosine_similarity_matrix(embedding_tensor)
  
  # Compute loss
  loss <- loss_fn(predicted_similarity, target_similarity)
  
  # Backpropagate
  loss$backward()
  
  # Update embeddings
  optimizer$step()
  
  # Print progress
  if (epoch %% 50 == 0) {
    cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
  }
}

# Extract reference embeddings (Fix: Use .detach() before conversion)
updated_embeddings <- embedding_tensor$detach()$cpu()$numpy()

# Print updated embeddings
print(updated_embeddings)



#### final product of updating /training
updated_embedding_tensor <- torch$tensor(updated_embeddings)




# Function to compute cosine similarity between all sentence pairs
cosine_similarity_matrix <- function(embeddings) {
  norm_emb <- embeddings / embeddings$norm(p = 2, dim = 1L, keepdim = TRUE)  
  return(norm_emb$mm(norm_emb$t()))  # Matrix multiplication for similarity
}

# Compute similarity matrix
updated_similarity_matrix <- cosine_similarity_matrix(updated_embedding_tensor)

# Convert tensor to matrix for readability
similarity_matrix_r <- updated_similarity_matrix$cpu()$detach()$numpy()

# Print similarity matrix
print(similarity_matrix_r)

cosine_similarity(similarity_matrix_r, target_matrix  )


#### final product of updating /training
## updated_embedding_tensor <- torch$tensor(updated_embeddings)

# Function to compute cosine similarity between all sentence pairs
cosine_similarity_matrix <- function(embeddings) {
  norm_emb <- embeddings / embeddings$norm(p = 2, dim = 1L, keepdim = TRUE)  
  return(norm_emb$mm(norm_emb$t()))  # Matrix multiplication for similarity
}




#### use updated_embeddings that relate each item at pre-defined correlates. 

n_rows <- embedding_tensor$shape[0]

# Reconstruct the list of row embeddings (back to reference_embeddings)
reference_embeddings_2 <- lapply(0:(n_rows - 1), function(i) {
  updated_embedding_tensor[as.integer(i), ]  # Python-style zero-based indexing
})
reference_embeddings_2 <- lapply(reference_embeddings_2, function(x) {
  as.numeric(x$cpu()$numpy())  # This gives you a clean R numeric vector
})



#### now we have updated reference_embeddings for the question items








###### text level analysis starts

text_embeddings_list <- lapply(sentence_token[[1]], function(sentence) {
  model$encode(sentence)  # Generate embedding for each sentence in the text
})

# Step 3: Define the cosine similarity function
cosine_similarity <- function(vec1, vec2) {
  sum(vec1 * vec2) / (sqrt(sum(vec1^2)) * sqrt(sum(vec2^2)))
}

# Each row represents a sentence from the input text, each column a reference sentence similarity
similarity_matrix <- matrix(NA, nrow = length(text_embeddings_list), ncol = length(reference_embeddings_2))

for (i in 1:length(text_embeddings_list)) {
  for (j in 1:length(reference_embeddings_2)) {
    similarity_matrix[i, j] <- cosine_similarity(text_embeddings_list[[i]], reference_embeddings_2[[j]])
  }
}


similarity_matrices_list <- list()


# Step 5: Loop through each text, tokenize into sentences, generate embeddings, and compute similarities
for (text_index in 1:length(sentence_token)) {
  # Tokenize the text into sentences
  
  # Generate embeddings for each sentence in the current text
  text_embeddings_list <- lapply(sentence_token[[text_index]], function(sentence) {
    model$encode(sentence)  # Generate embedding for each sentence
  })
  
  # Calculate the similarity matrix for the current text
  similarity_matrix <- matrix(NA, nrow = length(text_embeddings_list), ncol = length(reference_embeddings_2))
  
  for (i in 1:length(text_embeddings_list)) {
    for (j in 1:length(reference_embeddings_2)) {
      similarity_matrix[i, j] <- cosine_similarity(text_embeddings_list[[i]], reference_embeddings_2[[j]])
    }
  }
  
  # Store the similarity matrix in the list
  similarity_matrices_list[[text_index]] <- similarity_matrix
}

pad_similarity_matrices <- function(similarity_matrices_list) {
  # Find the maximum number of sentences across all matrices
  max_sentences <- max(sapply(similarity_matrices_list, function(matrix) dim(matrix)[1]))
  
  # Function to pad each matrix
  pad_matrix <- function(matrix, max_sentences) {
    # Get the number of sentences in the current matrix
    num_sentences <- dim(matrix)[1]
    
    # If the matrix already has max_sentences, return it as is
    if (num_sentences == max_sentences) {
      return(matrix)
    }
    
    # Create a padded matrix with zeros (or any other padding value)
    padded_matrix <- matrix(0, nrow = max_sentences, ncol = dim(matrix)[2])
    
    # Copy the original matrix into the padded matrix
    padded_matrix[1:num_sentences, ] <- matrix
    
    return(padded_matrix)
  }
  
  # Apply the padding function to all matrices in the list
  padded_similarity_matrices <- lapply(similarity_matrices_list, pad_matrix, max_sentences)
  
  return(padded_similarity_matrices)
}




#### padded_similarity_matrices_list has all transformed text
 
padded_similarity_matrices_list <- pad_similarity_matrices(similarity_matrices_list)
max_similarity_per_sentence <- lapply(padded_similarity_matrices_list, function(mat) {
  apply(mat, 1, max)
})

max_similarity_per_sentence <- lapply(padded_similarity_matrices_list, function(mat) {
  apply(mat, 1, max)  # This gives you a vector of length 201
})


dim(padded_similarity_matrices_list[[2]])
length(padded_similarity_matrices_list[[1]])

sentence_token[[1]][56]
similarity_matrices_list[[4]][[5]]

### pick 1 through 7
selected_dims <- c(1,2, 3, 4, 5, 6, 7)  # or 1:3, etc.

trimmed_similarity_list <- lapply(padded_similarity_matrices_list, function(mat) {
  mat[, selected_dims, drop = FALSE]  # Retain as matrix with selected columns
})


torch <- reticulate::import("torch")
nn <- torch$nn
F <- torch$nn$functional
input_data <- array(unlist(trimmed_similarity_list), 
                    dim = c(length(trimmed_similarity_list), 
                            dim(trimmed_similarity_list[[1]])[1], 
                            dim(trimmed_similarity_list[[1]])[2]))  


# Convert the 3D array into a PyTorch tensor
padded_similarity_matrices_tensor <- torch$tensor(input_data, dtype = torch$float32)
padded_similarity_matrices_tensor$shape






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
        output = self.classifier(last_hidden)
        return output, last_hidden
")


TextClassifier <- py$TextClassifier
py_int <- function(x) reticulate::py_eval(sprintf("int(%d)", x))

# -------------------------
# 2) Dimensions / labels
# -------------------------
shp <- as.integer(reticulate::iterate(padded_similarity_matrices_tensor$size()))
n <- shp[1]; max_sentences <- shp[2]; emb_dim_detected <- shp[3]
if (!exists("embedding_dim") || is.null(embedding_dim)) embedding_dim <- emb_dim_detected

# Raw labels (from your data)
y_vector_raw <- as.integer(as.character(data$reported))

# Map labels to 0..K-1 consistently for Torch, but keep a map to original if needed
lvl <- sort(unique(y_vector_raw))
map <- setNames(seq_along(lvl) - 1L, lvl)    # e.g., {3->0, 5->1, 7->2}
y_vector <- unname(map[as.character(y_vector_raw)])
num_classes <- length(lvl)

# Torch tensor of labels
y_tensor_full <- torch$tensor(y_vector)$to(dtype = torch$int64)

# -------------------------
# 3) Metric helpers
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
    # Binary: positive class assumed to be '1' in the 0..1 remap (adjust if needed)
    roc_obj <- pROC::roc(
      response  = factor(y_true, levels = c(0, 1)),
      predictor = y_prob[, 2],
      quiet = TRUE
    )
    as.numeric(pROC::auc(roc_obj))
  } else {
    # Multiclass: set column names to match response levels
    lev_chr <- as.character(0:(k - 1))
    y_prob_df <- as.data.frame(y_prob)
    colnames(y_prob_df) <- lev_chr
    
    mc <- pROC::multiclass.roc(
      response  = factor(y_true, levels = 0:(k - 1)),
      predictor = y_prob_df,
      quiet = TRUE
    )
    as.numeric(mc$auc)  # macro-averaged pairwise AUC (Hand & Till)
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
    idx_train_t  <- torch$tensor(train_idx0, dtype = torch$long)
    idx_test_t   <- torch$tensor(c(test_index0), dtype = torch$long)
    
    # slice data
    input_train  <- torch$index_select(padded_similarity_matrices_tensor, dim = 0L, index = idx_train_t)
    y_train      <- torch$index_select(y_tensor_full, dim = 0L, index = idx_train_t)
    
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
      out   <- model(input_train)
      logits <- out[[1]]
      loss  <- criterion(logits, y_train)
      optimizer$zero_grad(); loss$backward(); optimizer$step()
    }
    
    # test tensor (shape: 1 x max_sentences x embedding_dim)
    test_tensor <- torch$index_select(padded_similarity_matrices_tensor, dim = 0L, index = idx_test_t)
    
    # inference
    model$eval()
    with(torch$no_grad(), {
      out_test <- model(test_tensor)
      logits   <- out_test[[1]]                    # [1, K]
      probs    <- torch$softmax(logits, dim = 1L)  # [1, K]
      pred_cls <- torch$argmax(logits, dim = 1L)   # [1]
    })
    
    results_pred[i] <- as.integer(pred_cls$cpu()$numpy())
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
hidden_dim <- 2
lr <- 0.005

acc_vec <- numeric(trials)
f1_vec  <- numeric(trials)
auc_vec <- numeric(trials)

for (t in 1:trials) {
  cat("\n=== Trial", t, "===\n")
  res <- run_loo_once(seed = 1000 + t, epochs = epochs_per_trial, lr = lr, hidden_dim = hidden_dim)
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

cat(sprintf("\nAccuracy (mean ± SD): %.3f ± %.3f\n", acc_mean, acc_sd))
cat(sprintf("Macro-F1 (mean ± SD): %.3f ± %.3f\n", f1_mean,  f1_sd))
cat(sprintf("AUC (mean ± SD): %.3f ± %.3f\n",  auc_mean, auc_sd))

