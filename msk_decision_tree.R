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
# Dependencies: partykit, ggparty, ggplot2, scales, patchwork, rpart
# =============================================================================


# ── Install / load packages ───────────────────────────────────────────────────
install.packages(c("partykit", "ggparty", "ggplot2", "scales", "patchwork"))
library(partykit)   # Tree objects and node utilities
library(ggparty)    # ggplot2 extension for decision tree visualisation
library(ggplot2)    # Core plotting
library(scales)     # Axis label formatting (label_number)
library(patchwork)  # Plot composition (available if multi-panel layout needed)
library(rpart)      # Recursive partitioning — fits the regression tree


# ── Column selection & renaming ───────────────────────────────────────────────
# Narrow df_raw to the outcome and the six binary clinical predictors.
# Clean names avoid formula-quoting issues in rpart() and downstream code.
df <- df_raw %>%
  select(
    risk_score_12m,           # Outcome: continuous MSK risk score at 12 months
    has_pain_general,         # Binary: any general pain diagnosis
    has_conservative_care,    # Binary: conservative care utilisation flag
    has_pt,                   # Binary: physical therapy utilisation flag
    has_inj,                  # Binary: injection utilisation flag
    has_lifestyle_comorbidity,# Binary: lifestyle-related comorbidity present
    has_acute_care            # Binary: acute care utilisation flag
  )

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
# Fix seed for reproducible 80/20 split; test set used for out-of-sample eval.
set.seed(42)
train_idx <- sample(nrow(df), 0.8 * nrow(df))
train <- df[train_idx, ]
test  <- df[-train_idx, ]


# ── Fit regression tree ───────────────────────────────────────────────────────
# maxdepth = 4  : caps complexity to four levels, improving interpretability.
# minsplit  = 30: a node must have ≥ 30 observations before attempting a split.
# cp        = 0.005: complexity parameter — splits that don't improve R² by at
#             least 0.5 % are not attempted, preventing trivial branches.
tree_model <- rpart(
  risk_score_12m ~ .,
  data    = train,
  method  = "anova",
  control = rpart.control(maxdepth = 4, minsplit = 30, cp = 0.005)
)


# ── Prune via the 1-SE rule ───────────────────────────────────────────────────
# The 1-SE rule selects the simplest tree whose cross-validated error (xerror)
# is within one standard error of the minimum — a more conservative choice than
# simply picking the tree with the lowest xerror, reducing risk of overfitting.
cp_table  <- tree_model$cptable
min_idx   <- which.min(cp_table[, "xerror"])
threshold <- cp_table[min_idx, "xerror"] + cp_table[min_idx, "xstd"]
se_idx    <- min(which(cp_table[, "xerror"] <= threshold))
best_cp   <- cp_table[se_idx, "CP"]
tree_pruned <- prune(tree_model, cp = best_cp)


# ── Convert to partykit format ────────────────────────────────────────────────
# as.party() wraps the rpart object in the partykit representation required by
# ggparty for plotting; node data and split information are preserved.
party_tree <- as.party(tree_pruned)


# ── Build the ggparty visualisation ──────────────────────────────────────────
# Color palette: teal (low risk) → coral (high risk) gradient on histograms.
risk_low  <- "#2a9d8f"   # Low-risk end of histogram fill gradient
risk_high <- "#e76f51"   # High-risk end of histogram fill gradient
node_bg   <- "#1e293b"   # Dark navy background for inner (split) node labels
text_col  <- "#f8fafc"   # Near-white text on dark node labels

p <- ggparty(party_tree, terminal_space = 0.35) +

  # ── Tree edges ─────────────────────────────────────────────────────────────
  geom_edge(color = "#94a3b8", size = 0.7) +

  # Edge labels: show the Yes/No or factor-level break values at each branch.
  geom_edge_label(
    aes(label = breaks_label),
    color         = "#334155",
    size          = 3.2,
    fontface      = "bold",
    fill          = "#e2e8f0",
    label.padding = unit(0.2, "lines"),
    label.r       = unit(0.15, "lines")
  ) +

  # ── Inner node labels: show the splitting variable name ────────────────────
  geom_node_label(
    aes(label = splitvar),
    ids           = "inner",
    color         = text_col,
    fill          = node_bg,
    size          = 3.5,
    fontface      = "bold",
    label.padding = unit(0.4, "lines"),
    label.r       = unit(0.2, "lines"),
    label.colour  = "#475569"
  ) +

  # ── Terminal node plots: mini histogram of risk score distribution ──────────
  # Each leaf node embeds a small histogram so the reader can see not just the
  # mean risk score but the full distribution within that patient segment.
  geom_node_plot(
    gglist = list(
      geom_histogram(
        aes(x = risk_score_12m, fill = after_stat(x)),
        bins  = 15,
        color = "white",
        size  = 0.2
      ),
      scale_fill_gradient(low = risk_low, high = risk_high, guide = "none"),
      scale_x_continuous(labels = label_number(accuracy = 0.1)),
      scale_y_continuous(labels = NULL),
      theme_minimal(base_family = "sans"),
      theme(
        panel.grid  = element_blank(),
        axis.title  = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(size = 6, color = "#64748b"),
        plot.margin = margin(2, 2, 2, 2)
      )
    ),
    shared_axis_labels = TRUE,
    height = 0.25,
    width  = 0.9
  ) +

  # ── Terminal node labels: mean risk score (μ) for each leaf ────────────────
  # nodeapply() extracts the mean risk score from each terminal node's data
  # subset, and the result is formatted as "μ = X.XX" below the histogram.
  geom_node_label(
    aes(
      label = paste0("μ = ", round(nodeapply(party_tree,
        ids = nodeids(party_tree, terminal = TRUE),
        FUN = function(n) mean(n$data$risk_score_12m, na.rm = TRUE)
      )[[as.character(id)]], 2))
    ),
    ids           = "terminal",
    color         = "#1e293b",
    fill          = "#f1f5f9",
    size          = 3,
    fontface      = "italic",
    nudge_y       = -0.06,
    label.padding = unit(0.25, "lines"),
    label.r       = unit(0.15, "lines"),
    label.colour  = "#cbd5e1"
  ) +

  # ── Overall plot theme & titles ────────────────────────────────────────────
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "#f8fafc", color = NA),
    plot.title      = element_text(size = 18, face = "bold", color = "#0f172a",
                                   hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle   = element_text(size = 11, color = "#475569",
                                   hjust = 0.5, margin = margin(b = 16)),
    plot.caption    = element_text(size = 8, color = "#94a3b8", hjust = 1),
    plot.margin     = margin(20, 20, 20, 20)
  ) +
  labs(
    title    = "Predictors of 12-Month MSK Risk Score",
    subtitle = "Regression tree · each leaf shows risk score distribution and mean",
    caption  = "Pruned via 1-SE rule · trained on 80% holdout"
  )


# ── Export ────────────────────────────────────────────────────────────────────
# Output at 300 dpi for print-quality use in reports or presentations.
ggsave(
  "decision_tree_msk_ggparty.png",
  plot   = p,
  width  = 14,
  height = 9,
  dpi    = 300,
  bg     = "#f8fafc"
)
cat("Saved: decision_tree_msk_ggparty.png\n")
