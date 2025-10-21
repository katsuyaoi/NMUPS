load("wher eresults from LSTM_bridge")
acc_vec_b <- acc_vec 
f1_vec_b <- f1_vec
auc_vec_b <-  auc_vec

acc_mean_b <- mean(acc_vec_b); acc_sd_b <- sd(acc_vec_b)
f1_mean_b  <- mean(f1_vec_b);  f1_sd_b  <- sd(f1_vec_b)
auc_mean_b <- mean(auc_vec_b); auc_sd_b <- sd(auc_vec_b)

load("where results from LSTM_BERT ")

acc_mean <- mean(acc_vec); acc_sd <- sd(acc_vec)
f1_mean  <- mean(f1_vec);  f1_sd  <- sd(f1_vec)
auc_mean <- mean(auc_vec); auc_sd <- sd(auc_vec)

cat(sprintf("\n[bridge] Accuracy (mean ± SD): %.3f ± %.3f\n", acc_mean, acc_sd))
cat(sprintf("[bridge] Macro-F1 (mean ± SD): %.3f ± %.3f\n", f1_mean,  f1_sd))
cat(sprintf("[bridge] AUC (mean ± SD): %.3f ± %.3f\n",  auc_mean, auc_sd))

cat(sprintf("\n[S-BERT] Accuracy (mean ± SD): %.3f ± %.3f\n", acc_mean_b, acc_sd_b))
cat(sprintf("[S-BERT] Macro-F1 (mean ± SD): %.3f ± %.3f\n", f1_mean_b,  f1_sd_b))
cat(sprintf("[S-BERT] AUC (mean ± SD): %.3f ± %.3f\n",  auc_mean_b, auc_sd_b))


library(ggplot2)
library(reshape2)  # for melt()
library(extrafont)

df <- data.frame(
  Accuracy = c(acc_vec, acc_vec_b),
  MacroF1       = c(f1_vec, f1_vec_b),
  AUC      = c(auc_vec, auc_vec_b),
  Model    = rep(c("Bridge", "S-BERT"),
                 each = length(acc_vec))
)

df_long <- melt(df, id.vars = "Model",
                variable.name = "Metric",
                value.name = "Score")

## Plot density curves for each metric side-by-side
ggplot(df_long, aes(x = Score, color = Model, fill = Model)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~ Metric, scales = "free") +
  theme_minimal(base_size = 14) +
  labs(title = "",
       x = "Score", y = "Density") +
  theme(text = element_text(family ="Times New Roman"))

setwd("")
ggsave(
  filename = "plot_highres.png",
  plot = last_plot(),   # or replace with your ggplot object
  dpi = 300,            # resolution in dots per inch
  width = 8,            # width in inches
  height = 6,           # height in inches
  units = "in"
)

## Example: assume you already have vectors from 100 bootstrap trials:
# acc_bridge, f1_bridge, auc_bridge
# acc_sbert,  f1_sbert,  auc_sbert

# Put into a data frame
unpaired_summary <- function(x, y, name = "Metric", B = 10000L, seed = 123) {
  set.seed(seed)
  n1 <- length(x); n2 <- length(y)
  diffs <- replicate(B, mean(sample(x, n1, replace = TRUE)) - mean(sample(y, n2, replace = TRUE)))
  list(
    metric = name,
    mean_bridge = mean(x),
    mean_sbert  = mean(y),
    mean_diff   = mean(x) - mean(y),
    ci_95       = quantile(diffs, c(0.025, 0.975), names = FALSE),
    t_p         = t.test(x, y)$p.value,
    wilcox_p    = wilcox.test(x, y, exact = FALSE)$p.value
  )
}

acc_u <- unpaired_summary(acc_vec, acc_vec_b, "Accuracy")
f1_u  <- unpaired_summary(f1_vec,  f1_vec_b,  "Macro-F1")
auc_u <- unpaired_summary(auc_vec, auc_vec_b, "AUC")

print(acc_u); print(f1_u); print(auc_u)

cat(sprintf("Accuracy: Bridge %.3f vs S-BERT %.3f (Δ=%.3f, 95%% CI [%.3f, %.3f], p_t=%.4f, p_wilcox=%.4f)\n",
            acc_u$mean_bridge, acc_u$mean_sbert, acc_u$mean_diff,
            acc_u$ci_95[1], acc_u$ci_95[2], acc_u$t_p, acc_u$wilcox_p))



load("C:/Users/oikat/Box/temporary workspace/Emma Paul Katsu Project/bridge_environment_1000.RData")
avg_aligned

cat("\nProcrustes SS (alignment error) summary across 1000 seeds:\n")
print(summary(ss_vals))

cat("\nResubstitution accuracy across 1000 seeds (diagnostic, not generalization):\n")
print(summary(acc_diag))


library(zoo)
library(plotly)



### smoothing function 
smooth_matrix <- function(mat, k_sentence = 6, k_dim = 1) {
  # Smooth over sentences (rows)
  smoothed_sentences <- apply(mat, 2, function(col) rollmean(col, k_sentence, fill = "extend"))
  # Smooth over dimensions (columns)
  smoothed_final <- t(apply(smoothed_sentences, 1, function(row) rollmean(row, k_dim, fill = "extend")))
  return(smoothed_final)
}




# --- simple getter from LIST of matrices ---
get_mat_from_list <- function(L, idx, expect_dims = 7) {
  stopifnot(is.list(L), idx >= 1, idx <= length(L))
  m <- as.matrix(L[[idx]])
  # If orientation is dims x sentences, flip to sentences x dims
  if (ncol(m) != expect_dims && nrow(m) == expect_dims) m <- t(m)
  if (!is.numeric(m)) storage.mode(m) <- "double"
  m
}




i <- 6
print(tokenized_text[[i]])
j <- 7
print(tokenized_text[[j]])
k <- 8
print(tokenized_text[[k]])




i <- 1
print(tokenized_text[[i]])
j <- 2
print(tokenized_text[[j]])
k <- 3
print(tokenized_text[[k]])




i <- 17
print(tokenized_text[[i]])
j <- 25
print(tokenized_text[[j]])
k <- 27
print(tokenized_text[[k]])






# build smoothed matrices (no reticulate involved)
#matrix_i <- smooth_matrix(get_mat_from_list(padded_similarity_matrices_list, i), k_sentence = 9, k_dim = 4)
#matrix_j <- smooth_matrix(get_mat_from_list(padded_similarity_matrices_list, j), k_sentence = 9, k_dim = 4)
#matrix_k <- smooth_matrix(get_mat_from_list(padded_similarity_matrices_list, k), k_sentence = 9, k_dim = 4)



matrix_i <- smooth_matrix(embedding_array[i,,], k_sentence = 6, k_dim = 1)
matrix_j <- smooth_matrix(embedding_array[j,,], k_sentence = 6, k_dim = 1)
matrix_k <- smooth_matrix(embedding_array[k,,], k_sentence = 6, k_dim = 1)



dim_labels <- c("Effort", "Reward", "Fairness", "Pressure", "Obligation", "Discipline", "Coping")

# Transpose for correct orientation (sentence x dimension → x = sentence)


plot_ly(showscale = FALSE) %>%
  add_surface(
    z = ~t(matrix_i),
    opacity = 0.6,
    colorscale = "Reds",
    name = paste("Narrative", i)
  ) %>%
  add_surface(
    z = ~t(matrix_j),
    opacity = 0.6,
    colorscale = "Blues",
    name = paste("Narrative", j)
  ) %>%
  add_surface(
    z = ~t(matrix_k),
    opacity = 0.6,
    colorscale = "Greens",
    name = paste("Narrative", k)
  ) %>%
  layout(
    title = list(
      text = "Experientially Learning",
      font = list(family = "Times New Roman", size = 18)
    ),
    scene = list(
      xaxis = list(
        title = list(text = "Sentence Index", font = list(family = "Times New Roman", size = 14)),
        tickfont = list(family = "Times New Roman")
      ),
      yaxis = list(
        title = list(text = "", font = list(family = "Times New Roman", size = 14)),
        tickmode = "array",
        tickvals = 0:6,
        ticktext = dim_labels,
        tickfont = list(family = "Times New Roman")
      ),
      zaxis = list(
        title = list(text = "", font = list(family = "Times New Roman", size = 14)),
        tickfont = list(family = "Times New Roman")
      )
    ),
    font = list(family = "Times New Roman") # global fallback
  )



print(tokenized_text[[1]])
print(tokenized_text[[2]])
print(tokenized_text[[3]])
lapply(tokenized_text[4:16], print)
lapply(tokenized_text[17:28], print)