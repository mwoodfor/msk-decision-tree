# =============================================================================
# Decision Tree Analysis: 12-Month MSK Risk Score Predictors
# =============================================================================
# Purpose:  Fit, prune, and visualize a regression tree predicting a patient's
#           12-month musculoskeletal (MSK) risk score from six binary clinical
#           features covering pain, care utilization, and comorbidities.
#
# Inputs:   df_raw  – a data frame in the calling environment containing at
#                     minimum the seven columns selected below.
# Outputs:  decision_tree_msk_ggparty.png  (14 × 9 in, 300 dpi)
#
# Plot layout
#   Root node  : "All Patients" box (orange) — full-sample mean, 95% CI, n
#   Inner nodes: split variable name only (dark navy)
#   Edge labels: True / False + full-sample mean, 95% CI, n for that branch
#   Terminal nodes: removed (stats are already on the edges above them)
#
# All counts and statistics use the FULL dataset (df), not just the training
# split, so marketing sees real member counts for outreach planning.
#
# Dependencies: partykit, ggparty, ggplot2, scales, rpart, dplyr
# =============================================================================


# ── Install / load packages ───────────────────────────────────────────────────
install.packages(c("partykit", "ggparty", "ggplot2", "scales", "rpart", "dplyr"))
library(partykit)   # Tree objects and node utilities
library(ggparty)    # ggplot2 extension for decision tree visualisation
library(ggplot2)    # Core plotting
library(scales)     # Axis label formatting
library(rpart)      # Recursive partitioning — fits the regression tree
library(dplyr)      # Data wrangling


# ── Column selection & type coercion ─────────────────────────────────────────
# Narrow df_raw to the outcome and the six binary clinical predictors.
# Clean names avoid formula-quoting issues in rpart() and downstream code.
df <- df_raw %>%
  select(
    risk_score_12m,            # Outcome: continuous MSK risk score at 12 months
    has_pain_general,          # Binary: any general pain diagnosis
    has_conservative_care,     # Binary: conservative care utilisation flag
    has_pt,                    # Binary: physical therapy utilisation flag
    has_inj,                   # Binary: injection utilisation flag
    has_lifestyle_comorbidity, # Binary: lifestyle-related comorbidity present
    has_acute_care             # Binary: acute care utilisation flag
  ) %>%
  mutate(
    risk_score_12m = as.numeric(risk_score_12m)
  )

# Sanity check: warn if coercion introduced unexpected NAs.
na_count <- sum(is.na(df$risk_score_12m))
if (na_count > 0)
  warning(sprintf(
    "%d rows have NA risk_score_12m after coercion — check source data.",
    na_count
  ))

# Human-readable labels used only in plot annotations (not in the model).
label_map <- c(
  has_pain_general          = "Pain",
  has_conservative_care     = "Conservative Care",
  has_pt                    = "Physical Therapy",
  has_inj                   = "Injections",
  has_lifestyle_comorbidity = "Lifestyle Comorbidity",
  has_acute_care            = "Acute Care"
)


# ── Train / test split ────────────────────────────────────────────────────────
# 80/20 split; tree is fitted on train only to avoid data leakage.
# Full dataset (df) is used downstream for all displayed statistics.
set.seed(42)
train_idx <- sample(nrow(df), 0.8 * nrow(df))
train <- df[train_idx, ]
test  <- df[-train_idx, ]


# ── Fit regression tree ───────────────────────────────────────────────────────
tree_model <- rpart(
  risk_score_12m ~ .,
  data    = train,
  method  = "anova",
  control = rpart.control(maxdepth = 4, minsplit = 30, cp = 0.005)
)


# ── Prune via the 1-SE rule ───────────────────────────────────────────────────
cp_table  <- tree_model$cptable
min_idx   <- which.min(cp_table[, "xerror"])
threshold <- cp_table[min_idx, "xerror"] + cp_table[min_idx, "xstd"]
se_idx    <- min(which(cp_table[, "xerror"] <= threshold))
best_cp   <- cp_table[se_idx, "CP"]
tree_pruned <- prune(tree_model, cp = best_cp)


# ── Convert to partykit format ────────────────────────────────────────────────
party_tree <- as.party(tree_pruned)
party_tree$data$risk_score_12m <- as.numeric(as.character(party_tree$data$risk_score_12m))


# ── Route the FULL dataset through the pruned tree ────────────────────────────
# predict(..., type = "node") returns the terminal node ID each full-sample
# observation is routed to. Combined with the tree structure, this lets us
# compute accurate member counts at every node for marketing outreach.
#
# node_path() walks from the root to each terminal node and collects all
# ancestor node IDs, so we know which observations pass through each inner node.
df$node_id <- predict(tree_pruned, newdata = df, type = "node")

# Map rpart terminal node numbers to partykit node IDs.
# as.party() re-indexes nodes; the partykit terminal IDs are stored in
# nodeids(party_tree, terminal = TRUE) and correspond 1-to-1 with the sorted
# unique values returned by rpart's predict(..., type = "node").
rpart_terminal_ids  <- sort(unique(df$node_id))
party_terminal_ids  <- sort(nodeids(party_tree, terminal = TRUE))
node_id_map         <- setNames(party_terminal_ids, as.character(rpart_terminal_ids))
df$party_node_id    <- node_id_map[as.character(df$node_id)]

# For each partykit node, collect the set of full-sample row indices that pass
# through it (terminal nodes: exact assignment; inner nodes: union of all
# descendant terminal nodes' observations).
#
# nodeids(party_tree) returns ALL node IDs (inner + terminal).
# children_node() is not exported; we walk the tree with kids_node().
all_node_ids <- nodeids(party_tree)

# Build a lookup: node ID -> vector of row indices in df routed through it.
node_rows <- lapply(setNames(all_node_ids, all_node_ids), function(nid) {
  # Get all terminal descendants of this node (includes itself if terminal).
  desc_terminals <- nodeids(party_tree[[nid]], terminal = TRUE)
  # Map back to partykit IDs via the subtree's nodeids.
  # party_tree[[nid]] re-indexes from 1; add (nid - 1) to recover global IDs.
  # Simpler: just find which party_terminal_ids are descendants by checking
  # whether their rpart node IDs fall within the subtree.
  # Most robust approach: use the pre-computed df$party_node_id directly.
  which(df$party_node_id %in% nodeids(party_tree[[nid]], terminal = TRUE))
})


# ── Helper: compute stats for a vector of row indices ────────────────────────
node_stats <- function(rows) {
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

# Pre-compute stats for every node keyed by partykit node ID.
all_stats <- lapply(setNames(all_node_ids, all_node_ids), function(nid) {
  node_stats(node_rows[[nid]])
})

# Convenience named vectors for use inside aes() expressions.
stat_mean  <- sapply(all_stats, `[[`, "mean")
stat_ci_lo <- sapply(all_stats, `[[`, "ci_lo")
stat_ci_hi <- sapply(all_stats, `[[`, "ci_hi")
stat_n     <- sapply(all_stats, `[[`, "n")


# ── Full-sample root node annotation ─────────────────────────────────────────
# Computed from df (all 27,413 members) independently of the tree structure.
root_x    <- df$risk_score_12m[!is.na(df$risk_score_12m)]
root_n    <- length(root_x)
root_mean <- round(mean(root_x), 3)
root_se   <- sd(root_x) / sqrt(root_n)
root_t    <- qt(0.975, df = root_n - 1)
root_ci_lo <- round(root_mean - root_t * root_se, 3)
root_ci_hi <- round(root_mean + root_t * root_se, 3)

root_label <- paste0(
  "All Patients\n",
  "\u03bc = ", root_mean, "  [", root_ci_lo, ", ", root_ci_hi, "]\n",
  "n = ", formatC(root_n, format = "d", big.mark = ",")
)


# ── Color palette ─────────────────────────────────────────────────────────────
# BCBSMA brand colors:
#   Primary Blue  #0057A8   Light Blue  #00A3E0
#   Orange        #F47920   Dark Navy   #003366
node_bg   <- "#003366"   # inner split nodes
root_bg   <- "#F47920"   # root "All Patients" node — distinct orange
edge_fill <- "#E8F4FB"   # edge label background
edge_col  <- "#0057A8"   # edge label text
text_col  <- "#FFFFFF"


# ── Build the ggparty visualisation ──────────────────────────────────────────
p <- ggparty(party_tree, terminal_space = 0.15) +

  # ── Tree edges ───────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # ── Edge labels: True/False + full-sample stats for the child node ───────────
  # Each edge leads to a child node. We look up the child's partykit node ID
  # using the `id` aesthetic (which ggparty maps to the CHILD node for edges)
  # and retrieve the pre-computed full-sample stats from the named vectors.
  #
  # breaks_label gives "< 0.5" (False) or ">= 0.5" (True) for binary vars.
  # The multi-line label stacks True/False, then mean [CI], then n.
  geom_edge_label(
    aes(
      label = paste0(
        ifelse(grepl(">=", breaks_label), "True", "False"), "\n",
        "\u03bc = ", stat_mean[as.character(id)],
        "  [", stat_ci_lo[as.character(id)],
        ", ",  stat_ci_hi[as.character(id)], "]\n",
        "n = ", formatC(stat_n[as.character(id)], format = "d", big.mark = ",")
      )
    ),
    color         = edge_col,
    size          = 2.8,
    fontface      = "plain",
    fill          = edge_fill,
    label.padding = unit(0.35, "lines"),
    label.r       = unit(0.15, "lines")
  ) +

  # ── Root node: "All Patients" — orange, full-sample stats ───────────────────
  # The root node is node ID 1. We target it specifically with ids = 1 and
  # annotate it with the pre-computed root_label string. The orange fill
  # visually distinguishes it as the "before any split" starting point.
  geom_node_label(
    aes(label = root_label),
    ids           = 1L,
    color         = text_col,
    fill          = root_bg,
    size          = 3.2,
    fontface      = "bold",
    label.padding = unit(0.5, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = root_bg
  ) +

  # ── Inner (non-root) split nodes: variable name only ────────────────────────
  # These nodes show only the splitting variable name. Stats are on the edges,
  # so repeating them here would clutter the plot.
  # ids = "inner" targets ALL inner nodes including the root; we exclude node 1
  # by passing the explicit IDs of inner nodes that are not the root.
  geom_node_label(
    aes(label = ifelse(is.na(splitvar), "", label_map[splitvar])),
    ids           = setdiff(nodeids(party_tree, terminal = FALSE), 1L),
    color         = text_col,
    fill          = node_bg,
    size          = 3.5,
    fontface      = "bold",
    label.padding = unit(0.4, "lines"),
    label.r       = unit(0.2, "lines"),
    label.col     = node_bg
  ) +

  # ── Overall plot theme & titles ───────────────────────────────────────────────
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "#F0F8FF", color = NA),
    plot.title      = element_text(size = 18, face = "bold", color = "#003366",
                                   hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle   = element_text(size = 11, color = "#0057A8",
                                   hjust = 0.5, margin = margin(b = 16)),
    plot.caption    = element_text(size = 8,  color = "#00A3E0", hjust = 1),
    plot.margin     = margin(20, 20, 20, 20)
  ) +
  labs(
    title    = "Predictors of 12-Month MSK Risk Score",
    subtitle = paste0(
      "Regression tree \u00b7 edge labels show full-sample mean (95% CI) and member count",
      "\n",
      "Orange root = all ", formatC(root_n, format = "d", big.mark = ","),
      " members \u00b7 navy nodes = split variable"
    ),
    caption  = "Tree fitted on 80% training split, pruned via 1-SE rule \u00b7 statistics from full sample (n = 27,413)"
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
