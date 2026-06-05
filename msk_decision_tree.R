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
# Dependencies: partykit, ggparty, ggplot2, scales, rpart, dplyr, haven, janitor
# =============================================================================

# ── Load packages ─────────────────────────────────────────────────────────────
# install.packages(c("partykit","ggparty","ggplot2","scales","rpart","dplyr","haven","janitor"))
library(partykit)
library(ggparty)
library(ggplot2)
library(scales)
library(rpart)
library(haven)
library(dplyr)
library(janitor)


# ── Load & prepare data ───────────────────────────────────────────────────────
df_raw <- read_sas("C:/Users/mmarti06/OneDrive - BCBSMA/Evaluation (Debbie)/Project Work/Medicare Stars/msk_clustering/data/inference_12m_05292026.sas7bdat")

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
  warning(sprintf("%d rows have NA risk_score_12m after coercion.", na_count))

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


# ── Convert to partykit ───────────────────────────────────────────────────────
party_tree <- as.party(tree_pruned)
party_tree$data$risk_score_12m <- as.numeric(
  as.character(party_tree$data$risk_score_12m)
)


# ── Route FULL dataset through tree ──────────────────────────────────────────
# predict() returns the leaf mean (yval) for each observation.
# We match yvals back to rpart leaf node NUMBERS (rownames of frame),
# then build a direct rpart-node-number → partykit-node-ID map.
#
# Key insight from diagnostics:
#   rpart leaf nodes : 4  5  6  7   (rownames of frame where var == "<leaf>")
#   partykit term IDs: 3  4  6  7   (NOT the same numbering — cannot sort-match)
#
# The correct mapping uses as.party()'s internal nodeids():
#   as.party() walks the rpart tree depth-first and assigns sequential IDs.
#   We recover the mapping by extracting the yval from each partykit terminal
#   node's data and matching to the rpart leaf yvals.

# Step 1: get predicted yval for every full-sample member
df_preds <- predict(tree_pruned, newdata = df)

# Step 2: build rpart-leaf-yval → rpart-leaf-node-number lookup
leaf_frame      <- tree_pruned$frame[tree_pruned$frame$var == "<leaf>", ]
rpart_leaf_nums <- as.integer(rownames(leaf_frame))
# Map each observation's predicted yval to a rpart leaf node number
df_rpart_node <- rpart_leaf_nums[
  match(round(df_preds, 10), round(leaf_frame$yval, 10))
]

# Step 3: build rpart-node-number → partykit-node-ID map
# Each partykit terminal node stores the training data routed to it.
# Its mean risk_score_12m equals the corresponding rpart yval.
# We extract that mean for each partykit terminal to find the correspondence.
party_terminal_ids <- sort(nodeids(party_tree, terminal = TRUE))

party_terminal_ymeans <- sapply(party_terminal_ids, function(pid) {
  x <- as.numeric(as.character(party_tree[[pid]]$data$risk_score_12m))
  mean(x, na.rm = TRUE)
})

# For each rpart leaf, find the partykit terminal with the closest yval
# (they should match exactly since both are trained means on the same data)
rpart_to_party <- setNames(
  party_terminal_ids[
    sapply(leaf_frame$yval, function(y)
      which.min(abs(party_terminal_ymeans - y))
    )
  ],
  as.character(rpart_leaf_nums)
)

cat("rpart → partykit node mapping:\n")
print(rpart_to_party)

# Step 4: assign partykit terminal node ID to every full-sample row
df$party_node <- rpart_to_party[as.character(df_rpart_node)]
cat("NA party_node assignments (should be 0):", sum(is.na(df$party_node)), "\n")
cat("Distribution across terminal nodes:\n")
print(table(df$party_node))


# ── Collect row indices per node ──────────────────────────────────────────────
all_node_ids <- nodeids(party_tree)

node_rows <- setNames(
  lapply(all_node_ids, function(nid) {
    # Get all partykit terminal IDs that are descendants of this node
    desc_terminals <- nodeids(party_tree[[nid]], terminal = TRUE)
    which(df$party_node %in% desc_terminals)
  }),
  as.character(all_node_ids)
)

# Sanity check — print n per node
cat("\nFull-sample n per partykit node:\n")
print(sapply(node_rows, length))


# ── Stats helper ──────────────────────────────────────────────────────────────
calc_stats <- function(rows) {
  x  <- df$risk_score_12m[rows]
  x  <- x[!is.na(x)]
  n  <- length(x)
  if (n < 2) return(list(mean = NA, ci_lo = NA, ci_hi = NA, n = n))
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


# ── Pre-build display strings ─────────────────────────────────────────────────
fmt_stats <- function(nid) {
  s <- all_stats[[as.character(nid)]]
  if (is.na(s$mean)) return("n/a")
  paste0(
    "\u03bc = ", s$mean, "  [", s$ci_lo, ", ", s$ci_hi, "]\n",
    "n = ", formatC(s$n, format = "d", big.mark = ",")
  )
}

node_stat_strings <- setNames(
  sapply(all_node_ids, fmt_stats),
  as.character(all_node_ids)
)

# Split variable labels per node
node_splitvar_label <- setNames(
  sapply(all_node_ids, function(nid) {
    sv <- party_tree[[nid]]$node$split$varid
    if (is.null(sv) || length(sv) == 0 || is.na(sv)) return(NA_character_)
    varname <- names(party_tree$data)[sv]
    if (varname %in% names(label_map)) label_map[[varname]] else varname
  }),
  as.character(all_node_ids)
)

# Root node stats from full dataset
root_x     <- df$risk_score_12m[!is.na(df$risk_score_12m)]
root_n     <- length(root_x)
root_mean  <- round(mean(root_x), 3)
root_se    <- sd(root_x) / sqrt(root_n)
root_ci_lo <- round(root_mean - qt(0.975, df = root_n - 1) * root_se, 3)
root_ci_hi <- round(root_mean + qt(0.975, df = root_n - 1) * root_se, 3)

root_stat_str <- paste0(
  "\u03bc = ", root_mean, "  [", root_ci_lo, ", ", root_ci_hi, "]\n",
  "n = ", formatC(root_n, format = "d", big.mark = ",")
)

# Node type sets
terminal_ids  <- nodeids(party_tree, terminal = TRUE)
inner_ids     <- setdiff(all_node_ids, terminal_ids)
nonroot_inner <- setdiff(inner_ids, 1L)
nonroot_all   <- setdiff(all_node_ids, 1L)


# ── Colors ────────────────────────────────────────────────────────────────────
node_bg   <- "#003366"
root_bg   <- "#F47920"
edge_fill <- "#E8F4FB"
edge_col  <- "#0057A8"
text_col  <- "#FFFFFF"


# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggparty(
  party_tree,
  terminal_space = 0.15,
  add_vars = list(
    node_stat = function(data, node) {
      node_stat_strings[as.character(node$id)]
    },
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

  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # True / False direction labels
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
    shift         = 0.28
  ) +

  # Stats labels on each edge (child node stats)
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
    shift         = 0.68
  ) +

  # Root node: orange "All Patients" box
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

  # Non-root inner nodes: variable name only
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

  # Terminal nodes: invisible
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
