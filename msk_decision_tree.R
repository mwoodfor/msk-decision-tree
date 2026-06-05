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
# NOTE: as.party() can silently coerce the numeric outcome to a factor in node
# data slots. All downstream uses of risk_score_12m call as.numeric() explicitly
# as a guard — see node_means and geom_histogram aes() below.
party_tree <- as.party(tree_pruned)


# ── Pre-compute terminal node means ──────────────────────────────────────────
# nodeapply() cannot be called inside aes() because `id` is a ggplot aesthetic
# variable, not an R object available at expression-build time. Instead, build
# a named numeric vector keyed by node ID here, then look it up inside aes()
# using the vectorised `node_means[as.character(id)]` pattern.
terminal_ids <- nodeids(party_tree, terminal = TRUE)
node_means   <- setNames(
  sapply(terminal_ids, function(i)
    # as.numeric(as.character()) is required: as.party() stores the outcome as
    # a factor in node$data. Plain as.numeric() returns factor level codes
    # (integers like 6051), not the original probability values. Converting
    # factor → character → numeric recovers the true 0–1 probability.
    mean(as.numeric(as.character(party_tree[[i]]$data$risk_score_12m)), na.rm = TRUE)
  ),
  as.character(terminal_ids)
)


# ── Build the ggparty visualisation ──────────────────────────────────────────
# Color palette: Blue Cross Blue Shield of Massachusetts brand colors.
#   Primary Blue  #0057A8 — deep cobalt, dominant brand color
#   Light Blue    #00A3E0 — bright cyan-blue, used for CTAs and highlights
#   Orange        #F47920 — warm accent orange, used for buttons and callouts
#   Dark Navy     #003366 — deep background blue
#
# Histogram gradient runs BCBSMA light blue (low risk) → BCBSMA orange (high risk).
# Inner node labels use the dark navy background with white text.
risk_low  <- "#00A3E0"   # BCBSMA Light Blue — low-risk end of histogram gradient
risk_high <- "#F47920"   # BCBSMA Orange     — high-risk end of histogram gradient
node_bg   <- "#003366"   # BCBSMA Dark Navy  — inner node label background
text_col  <- "#FFFFFF"   # White text on dark node labels

p <- ggparty(party_tree, terminal_space = 0.35) +

  # ── Tree edges ─────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # Edge labels: show the Yes/No or factor-level break values at each branch.
  geom_edge_label(
    aes(label = breaks_label),
    color         = "#0057A8",
    size          = 3.2,
    fontface      = "bold",
    fill          = "#E8F4FB",
    label.padding = unit(0.2, "lines"),
    label.r       = unit(0.15, "lines")
  ) +

  # ── Inner node labels: show the splitting variable name ────────────────────
  # label_map translates raw column names (e.g. "has_pt") to readable labels
  # (e.g. "Physical Therapy"). ifelse guards nodes where splitvar is NA (roots
  # of single-node trees) though in practice all inner nodes have a splitvar.
  geom_node_label(
    aes(label = ifelse(is.na(splitvar), "",
                       label_map[splitvar])),
    ids           = "inner",
    color         = text_col,
    fill          = node_bg,
    size          = 3.5,
    fontface      = "bold",
    label.padding = unit(0.4, "lines"),
    label.r       = unit(0.2, "lines")
  ) +

  # ── Terminal node plots: mini histogram of risk score distribution ──────────
  # Each leaf node embeds a small histogram so the reader can see not just the
  # mean risk score but the full distribution within that patient segment.
  geom_node_plot(
    gglist = list(
      geom_histogram(
        # as.numeric(as.character()) is required — see node_means comment above.
        # Plain as.numeric() on a factor returns level codes, not probabilities.
        aes(x = as.numeric(as.character(risk_score_12m)), fill = after_stat(x)),
        bins      = 15,
        color     = "white",
        linewidth = 0.2
      ),
      scale_fill_gradient(low = risk_low, high = risk_high, guide = "none"),
      scale_x_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.01)),
      scale_y_continuous(labels = NULL),
      theme_minimal(base_family = "sans"),
      theme(
        panel.grid  = element_blank(),
        axis.title  = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(size = 6, color = "#0057A8"),
        plot.margin = margin(2, 2, 2, 2)
      )
    ),
    shared_axis_labels = TRUE,
    height = 0.25,
    width  = 0.9
  ) +

  # ── Terminal node labels: mean risk score (μ) for each leaf ────────────────
  # node_means is a named vector (keyed by node ID as character) pre-computed
  # above. Looking it up with node_means[as.character(id)] is safe inside aes()
  # because the entire vector is available in the enclosing environment.
  geom_node_label(
    aes(
      label = paste0("\u03bc = ", round(node_means[as.character(id)], 3))
    ),
    ids           = "terminal",
    color         = "#0057A8",
    fill          = "#E8F4FB",
    size          = 3,
    fontface      = "italic",
    nudge_y       = 0.06,    # positive = above the histogram, not overlapping it
    label.padding = unit(0.25, "lines"),
    label.r       = unit(0.15, "lines")
  ) +

  # ── Overall plot theme & titles ────────────────────────────────────────────
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
  bg     = "#F0F8FF"
)
cat("Saved: decision_tree_msk_ggparty.png\n")
