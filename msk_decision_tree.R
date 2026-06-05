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
#   Terminal nodes: invisible (stats are already on the incoming edge labels)
#
# All counts and statistics use the FULL dataset (df), not just the training
# split, so marketing sees real member counts for outreach planning.
#
# Key implementation note
#   add_vars in ggparty() injects per-node values into the NODE data frame,
#   which is available in geom_node_label() but NOT in geom_edge_label().
#   Edge labels reference the child node's id, but add_vars columns are only
#   accessible at node-draw time, not edge-draw time.
#   Solution: pre-compute all display strings as named character vectors keyed
#   by partykit node ID *outside* the plot call, then look them up inside
#   add_vars using the node id so they flow through to geom_node_label().
#   geom_edge_label() is used only for the True/False direction label;
#   the stats are placed via geom_node_label() nudged onto the edge midpoint.
#
# Dependencies: partykit, ggparty, ggplot2, scales, rpart, dplyr
# =============================================================================


# ── Install / load packages ───────────────────────────────────────────────────
install.packages(c("partykit", "ggparty", "ggplot2", "scales", "rpart", "dplyr"))
library(partykit)
library(ggparty)
library(ggplot2)
library(scales)
library(rpart)
library(dplyr)


# ── Column selection & type coercion ─────────────────────────────────────────
df <- df_raw %>%
  select(
    risk_score_12m,
    has_pain_general,
    has_conservative_care,
    has_pt,
    has_inj,
    has_lifestyle_comorbidity,
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
  has_pain_general          = "Pain",
  has_conservative_care     = "Conservative Care",
  has_pt                    = "Physical Therapy",
  has_inj                   = "Injections",
  has_lifestyle_comorbidity = "Lifestyle Comorbidity",
  has_acute_care            = "Acute Care"
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
# predict(..., type = "node") assigns each full-sample member to a terminal
# node using the rpart node numbering. We map these to partykit node IDs and
# then walk the tree to find which full-sample members pass through every node
# (terminal AND inner), giving accurate counts for marketing outreach.
df$rpart_node  <- predict(tree_pruned, newdata = df, type = "node")

rpart_terminals <- sort(unique(df$rpart_node))
party_terminals <- sort(nodeids(party_tree, terminal = TRUE))
rpart_to_party  <- setNames(party_terminals, as.character(rpart_terminals))
df$party_node   <- rpart_to_party[as.character(df$rpart_node)]

all_node_ids <- nodeids(party_tree)

# For each node, collect full-sample row indices passing through it.
# Terminal nodes: exact assignment. Inner nodes: union of descendant terminals.
node_rows <- setNames(
  lapply(all_node_ids, function(nid) {
    desc_terminals <- nodeids(party_tree[[nid]], terminal = TRUE)
    which(df$party_node %in% desc_terminals)
  }),
  as.character(all_node_ids)
)


# ── Helper: stats for a set of row indices ───────────────────────────────────
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
# These named vectors are looked up inside add_vars using the current node's id.
# This is the only reliable way to get per-node custom strings into both
# geom_node_label() (for the node boxes) and the edge stat labels (placed via
# a second geom_node_label() nudged to the edge midpoint — see plot section).

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

# Split variable display names keyed by node id (NA for terminal nodes)
node_splitvar_label <- setNames(
  sapply(all_node_ids, function(nid) {
    sv <- party_tree[[nid]]$node$split$varid
    if (is.null(sv) || is.na(sv)) return(NA_character_)
    varname <- names(party_tree$data)[sv]
    if (varname %in% names(label_map)) label_map[[varname]] else varname
  }),
  as.character(all_node_ids)
)

# Root node: full-sample stats (computed directly from df, not tree routing)
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

# Identify node types
terminal_ids  <- nodeids(party_tree, terminal = TRUE)
inner_ids     <- setdiff(all_node_ids, terminal_ids)
nonroot_inner <- setdiff(inner_ids, 1L)


# ── Color palette (BCBSMA brand) ──────────────────────────────────────────────
node_bg   <- "#003366"   # Dark Navy  — inner split nodes
root_bg   <- "#F47920"   # Orange     — root "All Patients" node
edge_fill <- "#E8F4FB"   # Light blue — edge label background
edge_col  <- "#0057A8"   # Primary blue — edge label text
text_col  <- "#FFFFFF"   # White text on dark backgrounds


# ── Build the ggparty plot ────────────────────────────────────────────────────
#
# Strategy for edge stats:
#   geom_edge_label() can only display a single-column aesthetic per edge.
#   It does NOT have access to add_vars columns.
#   Instead we use TWO geom_edge_label() calls:
#     1. True/False direction label (via breaks_label remap)
#     2. Stats label — injected by pre-computing a "child_stat" string in
#        add_vars (keyed from the node's own id, which IS available), then
#        using geom_node_label() nudged upward onto the incoming edge midpoint.
#   This is the standard workaround for ggparty's edge/node data separation.

p <- ggparty(
  party_tree,
  terminal_space = 0.15,
  add_vars = list(

    # For each node, look up the pre-computed stat string
    node_stat = function(data, node) {
      node_stat_strings[as.character(id(node))]
    },

    # Display label: "All Patients" for root, variable name for other inners,
    # empty for terminals
    node_header = function(data, node) {
      nid <- as.character(id(node))
      if (id(node) == 1L) {
        "All Patients"
      } else if (id(node) %in% terminal_ids) {
        ""
      } else {
        lbl <- node_splitvar_label[nid]
        if (is.na(lbl)) "" else lbl
      }
    },

    # Is this node the root?
    is_root = function(data, node) id(node) == 1L,

    # Is this node terminal?
    is_terminal = function(data, node) id(node) %in% terminal_ids
  )
) +

  # ── Tree edges ───────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # ── Edge direction labels: True / False ──────────────────────────────────────
  # parse = FALSE prevents ggparty from trying to parse our custom label as an
  # R expression (which would break the newline and bracket characters).
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
    shift         = 0.3    # place closer to parent node so stats fit below
  ) +

  # ── Edge stat labels: mean, CI, n — placed near midpoint via node nudge ──────
  # These are placed at each NON-ROOT node position and nudged upward along the
  # incoming edge. We target all non-root nodes (inner + terminal) so every
  # branch gets annotated. shift = 0.65 moves the label 65% of the way from
  # child toward parent, placing it on the lower half of each edge.
  geom_edge_label(
    aes(label = node_stat[as.character(id)]),
    ids           = setdiff(all_node_ids, 1L),   # all nodes except root
    parse         = FALSE,
    color         = edge_col,
    size          = 2.8,
    fontface      = "plain",
    fill          = edge_fill,
    label.padding = unit(0.3, "lines"),
    label.r       = unit(0.12, "lines"),
    label.size    = 0.3,
    shift         = 0.68   # lower portion of edge, below the True/False label
  ) +

  # ── Root node: "All Patients" (orange) ───────────────────────────────────────
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

  # ── Non-root inner split nodes: variable name only (dark navy) ───────────────
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

  # ── Terminal nodes: render as invisible point so ggparty doesn't draw boxes ──
  # ggparty always reserves space for terminal nodes. Setting fill and color
  # to the background color and using a tiny label makes them invisible.
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
