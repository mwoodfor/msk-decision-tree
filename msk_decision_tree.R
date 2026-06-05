# =============================================================================
# Decision Tree Analysis: 12-Month MSK Risk Score Predictors
# =============================================================================
# Dependencies: partykit, ggparty, ggplot2, scales, rpart, dplyr, haven, janitor
# =============================================================================

library(partykit); library(ggparty); library(ggplot2); library(scales)
library(rpart);    library(haven);   library(dplyr);   library(janitor)

# ── Load & prepare data ───────────────────────────────────────────────────────
df_raw <- read_sas("C:/Users/mmarti06/OneDrive - BCBSMA/Evaluation (Debbie)/Project Work/Medicare Stars/msk_clustering/data/inference_12m_05292026.sas7bdat")

df <- df_raw %>%
  select(risk_score_12m, has_pain_general, has_conservative_care,
         has_pt, has_inj, has_lifestyle_comorb, has_acute_care) %>%
  mutate(risk_score_12m = as.numeric(risk_score_12m))

label_map <- c(
  has_pain_general      = "Pain",
  has_conservative_care = "Conservative Care",
  has_pt                = "Physical Therapy",
  has_inj               = "Injections",
  has_lifestyle_comorb  = "Lifestyle Comorbidity",
  has_acute_care        = "Acute Care"
)

# ── Train / test split ────────────────────────────────────────────────────────
set.seed(42)
train_idx <- sample(nrow(df), 0.8 * nrow(df))
train <- df[train_idx, ]; test <- df[-train_idx, ]

# ── Fit & prune ───────────────────────────────────────────────────────────────
tree_model <- rpart(risk_score_12m ~ ., data = train, method = "anova",
                    control = rpart.control(maxdepth = 4, minsplit = 30, cp = 0.005))
cp_table  <- tree_model$cptable
min_idx   <- which.min(cp_table[, "xerror"])
threshold <- cp_table[min_idx, "xerror"] + cp_table[min_idx, "xstd"]
best_cp   <- cp_table[min(which(cp_table[, "xerror"] <= threshold)), "CP"]
tree_pruned <- prune(tree_model, cp = best_cp)

party_tree <- as.party(tree_pruned)
party_tree$data$risk_score_12m <- as.numeric(as.character(party_tree$data$risk_score_12m))

# ── Map every full-sample row to a partykit terminal node ────────────────────
df_preds   <- predict(tree_pruned, newdata = df)
leaf_frame <- tree_pruned$frame[tree_pruned$frame$var == "<leaf>", ]
rpart_nums <- as.integer(rownames(leaf_frame))

df_rpart_node <- rpart_nums[match(round(df_preds, 10), round(leaf_frame$yval, 10))]

party_terminal_ids <- sort(nodeids(party_tree, terminal = TRUE))
party_ymeans <- sapply(party_terminal_ids, function(pid) {
  mean(as.numeric(as.character(party_tree[[pid]]$data$risk_score_12m)), na.rm = TRUE)
})
rpart_to_party <- setNames(
  party_terminal_ids[sapply(leaf_frame$yval, function(y) which.min(abs(party_ymeans - y)))],
  as.character(rpart_nums)
)
df$party_node <- as.integer(rpart_to_party[as.character(df_rpart_node)])

# ── Build node → row-index lookup via rpart ancestor walk ────────────────────
all_rpart_nodes <- as.integer(rownames(tree_pruned$frame))
rpart_leaf_nums <- rpart_nums

rpart_parent <- function(n) as.integer(floor(n / 2))

rpart_terminals_of <- setNames(vector("list", length(all_rpart_nodes)),
                                as.character(all_rpart_nodes))
for (leaf in rpart_leaf_nums) {
  node <- leaf
  while (TRUE) {
    key <- as.character(node)
    rpart_terminals_of[[key]] <- c(rpart_terminals_of[[key]], leaf)
    if (node <= 1L) break
    node <- rpart_parent(node)
  }
}

rpart_train_n <- sapply(as.character(all_rpart_nodes), function(rn)
  tree_pruned$frame[rn, "n"])

all_party_ids <- nodeids(party_tree)

party_train_n <- sapply(all_party_ids, function(pid) nrow(party_tree[[pid]]$data))

party_to_rpart <- setNames(
  sapply(party_train_n, function(pn) {
    candidates <- names(rpart_train_n)[rpart_train_n == pn]
    if (length(candidates) == 1) as.integer(candidates) else NA_integer_
  }),
  as.character(all_party_ids)
)

node_rows <- setNames(
  lapply(all_party_ids, function(pid) {
    rn <- party_to_rpart[as.character(pid)]
    if (is.na(rn)) return(integer(0))
    leaves_rpart  <- rpart_terminals_of[[as.character(rn)]]
    party_leaves  <- as.integer(rpart_to_party[as.character(leaves_rpart)])
    which(df$party_node %in% party_leaves)
  }),
  as.character(all_party_ids)
)

cat("Full-sample n per partykit node:\n")
print(sapply(node_rows, length))

# ── Stats per node ────────────────────────────────────────────────────────────
calc_stats <- function(rows) {
  x <- df$risk_score_12m[rows]; x <- x[!is.na(x)]; n <- length(x)
  if (n < 2) return(list(mean = NA, ci_lo = NA, ci_hi = NA, n = n))
  m <- mean(x); se <- sd(x) / sqrt(n); tv <- qt(0.975, df = n - 1)
  list(mean  = round(m, 3),
       ci_lo = round(m - tv * se, 3),
       ci_hi = round(m + tv * se, 3),
       n     = n)
}

all_stats <- lapply(setNames(all_party_ids, as.character(all_party_ids)),
                    function(pid) calc_stats(node_rows[[as.character(pid)]]))

fmt_stats <- function(pid) {
  s <- all_stats[[as.character(pid)]]
  if (is.na(s$mean)) return("n/a")
  paste0("\u03bc = ", s$mean, "  [", s$ci_lo, ", ", s$ci_hi, "]\n",
         "n = ", formatC(s$n, format = "d", big.mark = ","))
}

node_stat_strings <- setNames(sapply(all_party_ids, fmt_stats),
                               as.character(all_party_ids))

# ── Split variable labels ─────────────────────────────────────────────────────
node_splitvar_label <- setNames(
  sapply(all_party_ids, function(pid) {
    sv <- party_tree[[pid]]$node$split$varid
    if (is.null(sv) || length(sv) == 0 || is.na(sv)) return(NA_character_)
    varname <- names(party_tree$data)[sv]
    if (varname %in% names(label_map)) label_map[[varname]] else varname
  }),
  as.character(all_party_ids)
)

# ── Root summary ──────────────────────────────────────────────────────────────
root_x      <- df$risk_score_12m[!is.na(df$risk_score_12m)]
root_n      <- length(root_x)
root_mean   <- round(mean(root_x), 3)
root_se     <- sd(root_x) / sqrt(root_n)
root_ci_lo  <- round(root_mean - qt(0.975, df = root_n - 1) * root_se, 3)
root_ci_hi  <- round(root_mean + qt(0.975, df = root_n - 1) * root_se, 3)
root_stat_str <- paste0("\u03bc = ", root_mean, "  [", root_ci_lo, ", ", root_ci_hi, "]\n",
                        "n = ", formatC(root_n, format = "d", big.mark = ","))

terminal_ids  <- nodeids(party_tree, terminal = TRUE)
inner_ids     <- setdiff(all_party_ids, terminal_ids)
nonroot_inner <- setdiff(inner_ids, 1L)
nonroot_all   <- setdiff(all_party_ids, 1L)

# ── Colors ────────────────────────────────────────────────────────────────────
node_bg   <- "#003366"; root_bg  <- "#F47920"
edge_fill <- "#E8F4FB"; edge_col <- "#0057A8"; text_col <- "#FFFFFF"

# ── Plot ──────────────────────────────────────────────────────────────────────
# FIX: use add_vars to inject pre-computed strings as named columns so they
# are always present in the ggparty data frame and can be referenced safely
# inside aes() without length-zero replacement errors.

p <- ggparty(party_tree,
             terminal_space = 0.15,
             add_vars = list(
               # stat label for this node (used on edge toward each child)
               stat_label = function(data, node) {
                 node_stat_strings[as.character(node$id)]
               },
               # split-variable header for inner nodes
               split_label = function(data, node) {
                 nid <- as.character(node$id)
                 if (node$id == 1L) return("All Patients")
                 lbl <- node_splitvar_label[nid]
                 if (is.na(lbl)) "" else lbl
               }
             )) +

  # ── Edges ──────────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # True / False branch labels (near the split node, shift ~ 0.25)
  geom_edge_label(
    aes(label = ifelse(grepl(">=", breaks_label), "True",
                       ifelse(grepl("<",  breaks_label), "False",
                              as.character(breaks_label)))),
    parse = FALSE,
    color = edge_col, size = 3.5, fontface = "bold",
    fill  = edge_fill,
    label.padding = unit(0.25, "lines"),
    label.r       = unit(0.12, "lines"),
    label.size    = 0.3,
    shift = 0.28) +

  # Stats label on every edge except the one from root (shift ~ 0.68)
  # stat_label is a column in the ggparty node data; ggparty copies node data
  # to edges so it is safely available here.
  geom_edge_label(
    aes(label = stat_label),
    ids   = nonroot_all,
    parse = FALSE,
    color = edge_col, size = 2.8, fontface = "plain",
    fill  = edge_fill,
    label.padding = unit(0.3,  "lines"),
    label.r       = unit(0.12, "lines"),
    label.size    = 0.3,
    shift = 0.68) +

  # ── Nodes ──────────────────────────────────────────────────────────────────
  # Root node (orange)
  geom_node_label(
    aes(label = paste0("All Patients\n", root_stat_str)),
    ids = 1L,
    color = text_col, fill = root_bg, size = 3.2, fontface = "bold",
    label.padding = unit(0.5, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = root_bg) +

  # Inner (non-root) split nodes (navy) — use split_label column
  geom_node_label(
    aes(label = split_label),
    ids = nonroot_inner,
    color = text_col, fill = node_bg, size = 3.5, fontface = "bold",
    label.padding = unit(0.4, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = node_bg) +

  # Terminal nodes — invisible placeholder so layout isn't disturbed
  geom_node_label(
    aes(label = ""),
    ids = terminal_ids,
    color = "#F0F8FF", fill = "#F0F8FF", size = 0.1,
    label.padding = unit(0.05, "lines"),
    label.r       = unit(0,    "lines"),
    label.size    = 0) +

  # ── Theme & labels ─────────────────────────────────────────────────────────
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "#F0F8FF", color = NA),
    plot.title    = element_text(size = 18, face = "bold", color = "#003366",
                                 hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle = element_text(size = 11, color = "#0057A8",
                                 hjust = 0.5, margin = margin(b = 16)),
    plot.caption  = element_text(size = 8,  color = "#00A3E0", hjust = 1),
    plot.margin   = margin(20, 20, 20, 20)) +
  labs(
    title    = "Predictors of 12-Month MSK Risk Score",
    subtitle = paste0(
      "Regression tree \u00b7 edge labels show full-sample mean (95% CI) and member count\n",
      "Orange root = all ", formatC(root_n, format = "d", big.mark = ","),
      " members \u00b7 navy nodes = split variable"),
    caption  = paste0(
      "Tree fitted on 80% training split, pruned via 1-SE rule",
      " \u00b7 statistics from full sample (n = ",
      formatC(root_n, format = "d", big.mark = ","), ")"))

ggsave("decision_tree_msk_ggparty.png", plot = p,
       width = 16, height = 10, dpi = 300, bg = "#F0F8FF")
cat("Saved: decision_tree_msk_ggparty.png\n")
