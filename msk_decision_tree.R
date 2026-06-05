# =============================================================================
# Decision Tree Analysis: 12-Month MSK Risk Score Predictors
# =============================================================================
# Purpose:  Fit, prune, and visualize a regression tree predicting a patient's
#           12-month musculoskeletal (MSK) risk score from six binary clinical
#           features covering pain, care utilization, and comorbidities.
#
# Inputs:   df_raw  – a data frame in the calling environment containing at
#                     minimum the seven columns selected below.
# Outputs:  decision_tree_msk_ggparty.png  (16 x 10 in, 300 dpi)
#
# Plot layout
#   Root node    : "All Patients" box (orange) — full-sample mean, 95% CI, n
#   Inner nodes  : split variable name only (dark navy)
#   Edge labels  : True / False + full-sample mean, 95% CI, n for child branch
#   Terminal nodes: invisible
#
# All counts and statistics use the FULL dataset (df), not just the training
# split, so marketing sees real member counts for outreach planning.
#
# Dependencies: partykit, ggparty, ggplot2, scales, rpart, dplyr, haven, janitor
# =============================================================================


# ── Install / load packages ───────────────────────────────────────────────────
# install.packages(c("partykit", "ggparty", "ggplot2", "scales", "rpart",
#                    "dplyr", "haven", "janitor"))
library(partykit)
library(ggparty)
library(ggplot2)
library(scales)
library(rpart)
library(haven)
library(dplyr)
library(janitor)


# ── Load data ─────────────────────────────────────────────────────────────────
df_raw <- read_sas("C:/Users/mmarti06/OneDrive - BCBSMA/Evaluation (Debbie)/Project Work/Medicare Stars/msk_clustering/data/inference_12m_05292026.sas7bdat")


# ── Column selection & type coercion ─────────────────────────────────────────
df <- df_raw %>%
  select(
    risk_score_12m,
    has_pain_general,
    has_conservative_care,
    has_pt,
    has_inj,
    has_lifestyle_comorb,
    has_acute_care
  ) %>%
  mutate(risk_score_12m = as.numeric(risk_score_12m))

na_count <- sum(is.na(df$risk_score_12m))
if (na_count > 0)
  warning(sprintf(
    "%d rows have NA risk_score_12m after coercion.", na_count
  ))

# Human-readable labels for split variable names
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
train <- df[train_idx, ]
test  <- df[-train_idx, ]


# ── Fit & prune regression tree ───────────────────────────────────────────────
tree_model <- rpart(
  risk_score_12m ~ .,
  data    = train,
  method  = "anova",
  control = rpart.control(maxdepth = 4, minsplit = 30, cp = 0.005)
)

cp_table  <- tree_model$cptable
min_idx   <- which.min(cp_table[, "xerror"])
threshold <- cp_table[min_idx, "xerror"] + cp_table[min_idx, "xstd"]
se_idx    <- min(which(cp_table[, "xerror"] <= threshold))
best_cp   <- cp_table[se_idx, "CP"]
tree_pruned <- prune(tree_model, cp = best_cp)


# ── Convert to partykit & patch numeric outcome ───────────────────────────────
party_tree <- as.party(tree_pruned)
party_tree$data$risk_score_12m <- as.numeric(
  as.character(party_tree$data$risk_score_12m)
)


# ── Route the FULL dataset through the pruned tree ────────────────────────────
# For regression trees (method = "anova"), rpart does not support
# predict(..., type = "node"). Instead, we use tree_pruned$where on the
# training set as a reference, and for new data we use rpart:::pred.rpart
# indirectly via predict() then match back to nodes.
#
# The reliable approach: apply the tree rules directly.
# rpart stores the frame with node numbers; we use rpart.predict.leaves()
# equivalent by passing the full df through predict() to get predicted values,
# then match each prediction to the node-level mean in tree_pruned$frame
# to recover the rpart node number for each observation.

# Step 1: get predicted value for every full-sample member
df_preds <- predict(tree_pruned, newdata = df)

# Step 2: the rpart frame contains the mean prediction ('yval') for each node.
# Leaf nodes are rows where var == "<leaf>". Match predicted value to leaf yval.
leaf_frame <- tree_pruned$frame[tree_pruned$frame$var == "<leaf>", ]
leaf_nodes  <- as.integer(rownames(leaf_frame))
leaf_yvals  <- leaf_frame$yval

# Each observation's predicted value equals exactly one leaf's yval.
# Use match() on rounded values to avoid floating-point mismatches.
df_rpart_node <- leaf_nodes[match(round(df_preds, 10), round(leaf_yvals, 10))]

# Step 3: map rpart leaf node numbers → partykit node IDs.
# as.party() preserves leaf ordering; party terminal IDs correspond 1-to-1
# with sorted rpart leaf node numbers.
rpart_terminals <- sort(unique(df_rpart_node))
party_terminals <- sort(nodeids(party_tree, terminal = TRUE))
rpart_to_party  <- setNames(party_terminals, as.character(rpart_terminals))
df$party_node   <- rpart_to_party[as.character(df_rpart_node)]

all_node_ids <- nodeids(party_tree)

# Step 4: for each partykit node, collect full-sample row indices routed
# through it. Terminal nodes: exact match. Inner nodes: union of all
# descendant terminal nodes' members.
node_rows <- setNames(
  lapply(all_node_ids, function(nid) {
    desc_terminals <- nodeids(party_tree[[nid]], terminal = TRUE)
    which(df$party_node %in% desc_terminals)
  }),
  as.character(all_node_ids)
)


# ── Helper: compute stats for a set of row indices ───────────────────────────
calc_stats <- function(rows) {
  x  <- df$risk_score_12m[rows]
  x  <- x[!is.na(x)]
  n  <- length(x)
  m  <- mean(x)
  se <- sd(x) / sqrt(n)
  t  <- qt(0.975, df = n - 1)
  list(
    mean  = round(m, 3),
    ci_lo = round(m - t * se, 3),
    ci_hi = round(m + t * se, 3),
    n     = n
  )
}

all_stats <- lapply(
  setNames(all_node_ids, as.character(all_node_ids)),
  function(nid) calc_stats(node_rows[[as.character(nid)]])
)


# ── Pre-build display strings keyed by partykit node ID ──────────────────────
# Named character vectors looked up in add_vars via node$id (not id(node),
# which conflicts with the deprecated dplyr::id() function).

fmt_stats <- function(nid) {
  s <- all_stats[[as.character(nid)]]
  paste0(
    "\u03bc = ", s$mean, "  [", s$ci_lo, ", ", s$ci_hi, "]\n",
    "n = ", formatC(s$n, format = "d", big.mark = ",")
  )
}

node_stat_strings <- setNames(
  sapply(all_node_ids, fmt_stats),
  as.character(all_node_ids)
)

# Split variable display name for each node (NA for terminals)
node_splitvar_label <- setNames(
  sapply(all_node_ids, function(nid) {
    sv <- party_tree[[nid]]$node$split$varid
    if (is.null(sv) || length(sv) == 0 || is.na(sv)) return(NA_character_)
    varname <- names(party_tree$data)[sv]
    if (varname %in% names(label_map)) label_map[[varname]] else varname
  }),
  as.character(all_node_ids)
)

# Root node stats from full sample (all df, independent of tree routing)
root_x     <- df$risk_score_12m[!is.na(df$risk_score_12m)]
root_n     <- length(root_x)
root_mean  <- round(mean(root_x), 3)
root_se    <- sd(root_x) / sqrt(root_n)
root_t     <- qt(0.975, df = root_n - 1)
root_ci_lo <- round(root_mean - root_t * root_se, 3)
root_ci_hi <- round(root_mean + root_t * root_se, 3)

root_stat_str <- paste0(
  "\u03bc = ", root_mean, "  [", root_ci_lo, ", ", root_ci_hi, "]\n",
  "n = ", formatC(root_n, format = "d", big.mark = ",")
)

# Node type sets
terminal_ids  <- nodeids(party_tree, terminal = TRUE)
inner_ids     <- setdiff(all_node_ids, terminal_ids)
nonroot_inner <- setdiff(inner_ids, 1L)
nonroot_all   <- setdiff(all_node_ids, 1L)


# ── Color palette (BCBSMA brand) ──────────────────────────────────────────────
node_bg   <- "#003366"   # Dark Navy  — inner split nodes
root_bg   <- "#F47920"   # Orange     — root "All Patients" node
edge_fill <- "#E8F4FB"   # Light blue — edge label background
edge_col  <- "#0057A8"   # Primary blue — edge label text
text_col  <- "#FFFFFF"   # White text on dark backgrounds


# ── Build the ggparty plot ────────────────────────────────────────────────────
# NOTE: inside add_vars functions, use node$id instead of id(node).
# dplyr::id() is deprecated and intercepts the partykit id() call, causing
# the "id() is defunct" error seen when dplyr is loaded alongside partykit.

p <- ggparty(
  party_tree,
  terminal_space = 0.15,
  add_vars = list(

    # Pre-computed stat string for this node (used in edge stat label)
    node_stat = function(data, node) {
      node_stat_strings[as.character(node$id)]
    },

    # Split variable display name for this node
    node_header = function(data, node) {
      nid <- as.character(node$id)
      if (node$id == 1L) {
        "All Patients"
      } else if (node$id %in% terminal_ids) {
        ""
      } else {
        lbl <- node_splitvar_label[nid]
        if (is.na(lbl)) "" else lbl
      }
    }

  )
) +

  # ── Tree edges ───────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # ── Edge direction labels: True / False ──────────────────────────────────────
  # parse = FALSE: prevents ggparty from interpreting our label as an R
  # expression, which would corrupt the text.
  geom_edge_label(
    aes(
      label = ifelse(
        grepl(">=", breaks_label), "True",
        ifelse(grepl("<", breaks_label), "False", as.character(breaks_label))
      )
    ),
    parse         = FALSE,
    color         = edge_col,
    size          = 3.5,
    fontface      = "bold",
    fill          = edge_fill,
    label.padding = unit(0.25, "lines"),
    label.r       = unit(0.12, "lines"),
    label.size    = 0.3,
    shift         = 0.28   # closer to parent node; stats label sits below this
  ) +

  # ── Edge stat labels: mean (95% CI) and n ────────────────────────────────────
  # geom_edge_label() does not have access to add_vars. The workaround:
  # use a second geom_edge_label() and reference node_stat via the `id`
  # aesthetic — which in edge context maps to the CHILD node id, matching
  # the keys in our node_stat_strings named vector.
  geom_edge_label(
    aes(label = node_stat_strings[as.character(id)]),
    ids           = nonroot_all,
    parse         = FALSE,
    color         = edge_col,
    size          = 2.8,
    fontface      = "plain",
    fill          = edge_fill,
    label.padding = unit(0.3, "lines"),
    label.r       = unit(0.12, "lines"),
    label.size    = 0.3,
    shift         = 0.68   # lower half of edge, below the True/False label
  ) +

  # ── Root node: "All Patients" orange box ─────────────────────────────────────
  geom_node_label(
    aes(label = paste0("All Patients\n", root_stat_str)),
    ids           = 1L,
    color         = text_col,
    fill          = root_bg,
    size          = 3.2,
    fontface      = "bold",
    label.padding = unit(0.5, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = root_bg
  ) +

  # ── Non-root inner nodes: split variable name only ───────────────────────────
  geom_node_label(
    aes(label = ifelse(
      is.na(node_splitvar_label[as.character(id)]),
      "",
      node_splitvar_label[as.character(id)]
    )),
    ids           = nonroot_inner,
    color         = text_col,
    fill          = node_bg,
    size          = 3.5,
    fontface      = "bold",
    label.padding = unit(0.4, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = node_bg
  ) +

  # ── Terminal nodes: invisible (background-coloured, no border) ───────────────
  geom_node_label(
    aes(label = ""),
    ids           = terminal_ids,
    color         = "#F0F8FF",
    fill          = "#F0F8FF",
    size          = 0.1,
    label.padding = unit(0.05, "lines"),
    label.r       = unit(0, "lines"),
    label.size    = 0
  ) +

  # ── Theme & titles ───────────────────────────────────────────────────────────
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "#F0F8FF", color = NA),
    plot.title      = element_text(size = 18, face = "bold", color = "#003366",
                                   hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle   = element_text(size = 11, color = "#0057A8",
                                   hjust = 0.5, margin = margin(b = 16)),
    plot.caption    = element_text(size = 8, color = "#00A3E0", hjust = 1),
    plot.margin     = margin(20, 20, 20, 20)
  ) +
  labs(
    title    = "Predictors of 12-Month MSK Risk Score",
    subtitle = paste0(
      "Regression tree \u00b7 edge labels show full-sample mean (95% CI) and member count\n",
      "Orange root = all ", formatC(root_n, format = "d", big.mark = ","),
      " members \u00b7 navy nodes = split variable"
    ),
    caption  = paste0(
      "Tree fitted on 80% training split, pruned via 1-SE rule",
      " \u00b7 statistics from full sample (n = ",
      formatC(root_n, format = "d", big.mark = ","), ")"
    )
  )


# ── Export ────────────────────────────────────────────────────────────────────
ggsave(
  "decision_tree_msk_ggparty.png",
  plot   = p,
  width  = 16,
  height = 10,
  dpi    = 300,
  bg     = "#F0F8FF"
)
cat("Saved: decision_tree_msk_ggparty.png\n")
