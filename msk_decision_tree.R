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
# Dependencies: partykit, ggparty, ggplot2, scales, rpart
# =============================================================================


# ── Install / load packages ───────────────────────────────────────────────────
install.packages(c("partykit", "ggparty", "ggplot2", "scales", "rpart"))
library(partykit)   # Tree objects and node utilities
library(ggparty)    # ggplot2 extension for decision tree visualisation
library(ggplot2)    # Core plotting
library(scales)     # Axis label formatting
library(rpart)      # Recursive partitioning — fits the regression tree


# ── Column selection & renaming ───────────────────────────────────────────────
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
    # risk_score_12m arrives as character in the source data; coerce to numeric
    # here so rpart(), partykit, and all plot code receive the correct type.
    risk_score_12m = as.numeric(risk_score_12m)
  )

# Sanity check: warn if coercion introduced unexpected NAs (e.g. non-numeric
# strings like "N/A" or "" present in the source data).
na_count <- sum(is.na(df$risk_score_12m))
if (na_count > 0)
  warning(sprintf(
    "%d rows have NA risk_score_12m after coercion — check source data for non-numeric values.",
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
# Fix seed for reproducible 80/20 split; test set used for out-of-sample eval.
set.seed(42)
train_idx <- sample(nrow(df), 0.8 * nrow(df))
train <- df[train_idx, ]
test  <- df[-train_idx, ]


# ── Fit regression tree ───────────────────────────────────────────────────────
# maxdepth = 4  : caps complexity to four levels, improving interpretability.
# minsplit  = 30: a node must have >= 30 observations before attempting a split.
# cp        = 0.005: complexity parameter — splits that don't improve R^2 by at
#             least 0.5% are not attempted, preventing trivial branches.
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

# as.party() can re-encode the numeric outcome as a factor in party_tree$data.
# A single coercion on the top-level $data slot fixes it for all nodes at once.
party_tree$data$risk_score_12m <- as.numeric(as.character(party_tree$data$risk_score_12m))


# ── Build the ggparty visualisation ──────────────────────────────────────────
# Color palette: Blue Cross Blue Shield of Massachusetts brand colors.
#   Primary Blue  #0057A8 — deep cobalt, dominant brand color
#   Light Blue    #00A3E0 — bright cyan-blue, used for CTAs and highlights
#   Orange        #F47920 — warm accent orange, used for buttons and callouts
#   Dark Navy     #003366 — deep background blue
node_bg  <- "#003366"   # BCBSMA Dark Navy  — inner node label background
text_col <- "#FFFFFF"   # White text on dark node labels

# ── add_vars: compute mean, 95% CI bounds, and n for every node ───────────────
# ggparty's add_vars argument accepts named functions of the form
# function(data, node). They are evaluated once per node at plot-build time,
# making the results available as columns inside any aes() call.
# qt(0.975, df) gives the t critical value for a two-sided 95% CI.
# The as.numeric(as.character()) coercion is retained as a safety net because
# as.party() may store risk_score_12m as a factor in per-node $data slots even
# after the top-level $data patch above.
p <- ggparty(
  party_tree,
  terminal_space = 0.2,
  add_vars = list(

    node_mean = function(data, node) {
      x <- as.numeric(as.character(node$data$risk_score_12m))
      round(mean(x, na.rm = TRUE), 3)
    },

    node_ci_lo = function(data, node) {
      x  <- as.numeric(as.character(node$data$risk_score_12m))
      n  <- sum(!is.na(x))
      se <- sd(x, na.rm = TRUE) / sqrt(n)
      round(mean(x, na.rm = TRUE) - qt(0.975, df = n - 1) * se, 3)
    },

    node_ci_hi = function(data, node) {
      x  <- as.numeric(as.character(node$data$risk_score_12m))
      n  <- sum(!is.na(x))
      se <- sd(x, na.rm = TRUE) / sqrt(n)
      round(mean(x, na.rm = TRUE) + qt(0.975, df = n - 1) * se, 3)
    },

    node_n = function(data, node) {
      sum(!is.na(as.numeric(as.character(node$data$risk_score_12m))))
    }

  )
) +

  # ── Tree edges ───────────────────────────────────────────────────────────────
  geom_edge(color = "#00A3E0", linewidth = 0.7) +

  # ── Edge labels: True / False instead of >= 0.5 / < 0.5 ────────────────────
  # All predictors are binary (0/1). rpart splits them at 0.5, producing the
  # breaks_label values "< 0.5" (i.e. 0 = False) and ">= 0.5" (i.e. 1 = True).
  # The ifelse() remaps these to human-readable True/False labels.
  geom_edge_label(
    aes(
      label = ifelse(
        grepl(">=", breaks_label), "True",
        ifelse(grepl("<",  breaks_label), "False", breaks_label)
      )
    ),
    color         = "#0057A8",
    size          = 3.2,
    fontface      = "bold",
    fill          = "#E8F4FB",
    label.padding = unit(0.2, "lines"),
    label.r       = unit(0.15, "lines")
  ) +

  # ── Inner node labels: variable name + mean (95% CI) + n ────────────────────
  # geom_node_label()'s line_list argument accepts one aes() per line of text;
  # line_gpar sets font size, color, and style for each line independently.
  # The three stats (mean, CI, n) are pre-computed via add_vars above and are
  # available here by their names as ggparty data columns.
  geom_node_label(
    ids       = "inner",
    line_list = list(
      # Line 1: human-readable variable name from label_map
      aes(label = ifelse(is.na(splitvar), "", label_map[splitvar])),
      # Line 2: mean risk score with 95% confidence interval
      aes(label = paste0("\u03bc = ", node_mean,
                         "  [", node_ci_lo, ", ", node_ci_hi, "]")),
      # Line 3: sample size at this node
      aes(label = paste0("n = ", formatC(node_n, format = "d", big.mark = ",")))
    ),
    line_gpar = list(
      list(size = 10, col = text_col, fontface = "bold"),   # variable name
      list(size =  8, col = "#A8C8E8", fontface = "plain"), # mean + CI
      list(size =  8, col = "#A8C8E8", fontface = "plain")  # n
    ),
    fill          = node_bg,
    label.padding = unit(0.45, "lines"),
    label.r       = unit(0.2,  "lines"),
    label.col     = node_bg
  ) +

  # ── Terminal node labels: mean (95% CI) + n ──────────────────────────────────
  # Terminal nodes receive the same stats layout but without a variable name,
  # styled in BCBSMA light blue on a pale background for visual contrast.
  geom_node_label(
    ids       = "terminal",
    line_list = list(
      aes(label = paste0("\u03bc = ", node_mean,
                         "  [", node_ci_lo, ", ", node_ci_hi, "]")),
      aes(label = paste0("n = ", formatC(node_n, format = "d", big.mark = ",")))
    ),
    line_gpar = list(
      list(size = 8, col = "#0057A8", fontface = "plain"),
      list(size = 8, col = "#0057A8", fontface = "plain")
    ),
    fill          = "#E8F4FB",
    label.padding = unit(0.4,  "lines"),
    label.r       = unit(0.15, "lines"),
    label.col     = "#BDD9EF"
  ) +

  # ── Overall plot theme & titles ───────────────────────────────────────────────
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
    subtitle = "Regression tree \u00b7 each node shows mean risk score (95% CI) and sample size",
    caption  = "Pruned via 1-SE rule \u00b7 trained on 80% holdout"
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
