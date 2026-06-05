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
# predict() returns the leaf yval for each row.
# Match yvals → rpart leaf row numbers → partykit terminal IDs.
df_preds    <- predict(tree_pruned, newdata = df)
leaf_frame  <- tree_pruned$frame[tree_pruned$frame$var == "<leaf>", ]
rpart_nums  <- as.integer(rownames(leaf_frame))

# Assign each row a rpart leaf number
df_rpart_node <- rpart_nums[match(round(df_preds, 10), round(leaf_frame$yval, 10))]

# Match rpart leaf → partykit terminal via training-data means
party_terminal_ids <- sort(nodeids(party_tree, terminal = TRUE))
party_ymeans <- sapply(party_terminal_ids, function(pid) {
  mean(as.numeric(as.character(party_tree[[pid]]$data$risk_score_12m)), na.rm = TRUE)
})
rpart_to_party <- setNames(
  party_terminal_ids[sapply(leaf_frame$yval, function(y) which.min(abs(party_ymeans - y)))],
  as.character(rpart_nums)
)
df$party_node <- as.integer(rpart_to_party[as.character(df_rpart_node)])

cat("Node mapping:\n"); print(rpart_to_party)
cat("Assignments (NA should be 0):", sum(is.na(df$party_node)), "\n")
cat("Counts per terminal:\n"); print(table(df$party_node))

# ── Build ancestor lookup: which terminal IDs descend from each node ----------
# Use the rpart frame's node numbering structure directly.
# In rpart, children of node k are 2k (left/False) and 2k+1 (right/True).
# We walk this to find all terminal descendants of every node in the rpart tree,
# then translate to partykit IDs using rpart_to_party.
all_rpart_nodes <- as.integer(rownames(tree_pruned$frame))

get_rpart_terminal_descendants <- function(node, all_nodes, leaf_nodes) {
  if (node %in% leaf_nodes) return(node)
  left  <- node * 2L
  right <- node * 2L + 1L
  desc  <- c()
  if (left  %in% all_nodes) desc <- c(desc, get_rpart_terminal_descendants(left,  all_nodes, leaf_nodes))
  if (right %in% all_nodes) desc <- c(desc, get_rpart_terminal_descendants(right, all_nodes, leaf_nodes))
  desc
}

rpart_leaf_nums <- rpart_nums  # already computed above

# For every rpart node, get its terminal descendants (as rpart numbers)
rpart_desc_terminals <- setNames(
  lapply(all_rpart_nodes, function(n)
    get_rpart_terminal_descendants(n, all_rpart_nodes, rpart_leaf_nums)
  ),
  as.character(all_rpart_nodes)
)

# Translate to partykit terminal IDs
party_desc_terminals <- lapply(rpart_desc_terminals, function(rleaves)
  as.integer(rpart_to_party[as.character(rleaves)])
)

# For partykit nodes, build the same lookup by matching via yval
# We need: for each partykit node id, which partykit terminal IDs descend from it?
# Approach: each partykit node corresponds to an rpart node.
# Map partykit node → rpart node via yval matching on inner nodes too.

# Actually simplest: use df$party_node directly.
# For each partykit node id, find which rpart node it corresponds to,
# then use party_desc_terminals.

# Build partykit-node → rpart-node map
# For terminal nodes we already have rpart_to_party (inverted).
party_to_rpart_terminal <- setNames(
  as.integer(names(rpart_to_party)),
  as.character(rpart_to_party)
)

# For inner partykit nodes, match by finding which rpart inner node has the
# same set of terminal descendants.
all_party_ids <- nodeids(party_tree)

# Build node_rows: partykit node id → row indices in df
# For terminal partykit nodes: rows where df$party_node == nid
# For inner partykit nodes: rows where df$party_node is any terminal descendant
#
# We determine terminal descendants of each partykit inner node by:
# converting the partykit node's subtree terminal IDs.
# Since nodeids(party_tree[[nid]], terminal=TRUE) gives LOCAL ids, we instead
# use the known terminal partykit IDs and the tree structure directly:
# partykit node 1 contains all; node 2 (left of root) contains terminals {3,4};
# node 5 (right of root) contains terminals {6,7} — we read this from the tree.

# Robust approach: build subtree terminal membership from df$party_node.
# For each inner node, its members = all rows whose terminal is reachable
# by walking the party_tree kids structure.

get_party_terminal_descendants <- function(party_tree, nid) {
  node <- party_tree[[nid]]$node
  if (is.null(node$kids) || length(node$kids) == 0) {
    return(nid)  # terminal
  }
  kid_ids <- sapply(node$kids, function(k) k$id)
  # kid ids are LOCAL to the subtree — add offset
  # Actually party_tree[[nid]] re-indexes from 1.
  # Use nodeids() on the full tree to get global kids.
  # Safer: use the ggparty plot data which has parent column.
  NULL
}

# SIMPLEST CORRECT APPROACH:
# Build a flat data frame: one row per (partykit_node, partykit_terminal) pair
# indicating that terminal is a descendant of that node.
# We do this by walking from each terminal up to the root using the parent
# column from get_plot_data (which ggparty builds internally).
# We reconstruct the parent chain from the rpart frame structure.

# rpart parent of node k = floor(k/2), except root=1 has no parent.
rpart_parent <- function(n) as.integer(floor(n / 2))

# For each rpart terminal, walk up to root collecting ancestors
rpart_ancestors <- lapply(rpart_leaf_nums, function(leaf) {
  chain <- c(leaf)
  node  <- leaf
  while (node > 1L) {
    node  <- rpart_parent(node)
    chain <- c(chain, node)
  }
  chain
})

# Build: rpart_node → set of rpart terminal descendants
rpart_terminals_of <- setNames(
  vector("list", length(all_rpart_nodes)),
  as.character(all_rpart_nodes)
)
for (i in seq_along(rpart_leaf_nums)) {
  leaf <- rpart_leaf_nums[i]
  for (anc in rpart_ancestors[[i]]) {
    rpart_terminals_of[[as.character(anc)]] <- c(
      rpart_terminals_of[[as.character(anc)]], leaf
    )
  }
}

# Now build partykit node → row indices using rpart ancestor structure.
# We need to know which rpart node each partykit node corresponds to.
# For terminals: rpart_to_party gives rpart→party; invert to party→rpart.
# For inner nodes: partykit node 1 = rpart node 1 (root always matches).
# Build the full map by matching subtree sizes.

# partykit inner nodes and their rpart equivalents via member-set matching:
# Count members of each rpart inner node (from df_rpart_node)
rpart_node_n <- sapply(as.character(all_rpart_nodes), function(rn) {
  leaves <- rpart_terminals_of[[rn]]
  sum(df_rpart_node %in% leaves)
})

# Count members of each partykit node via terminal descendants
# For partykit terminals we know the counts. For inner nodes we need structure.
# Use: partykit node n contains the training rows in party_tree[[n]]$data
# That gives training-set sizes. Match to rpart node training sizes.
rpart_train_n <- sapply(as.character(all_rpart_nodes), function(rn) {
  tree_pruned$frame[rn, "n"]
})

party_train_n <- sapply(all_party_ids, function(pid) {
  nrow(party_tree[[pid]]$data)
})

# Match partykit node → rpart node by training n (unique within tree)
party_to_rpart <- setNames(
  sapply(party_train_n, function(pn) {
    candidates <- names(rpart_train_n)[rpart_train_n == pn]
    if (length(candidates) == 1) as.integer(candidates) else NA_integer_
  }),
  as.character(all_party_ids)
)

cat("\nPartykit → rpart node map (by training n):\n")
print(party_to_rpart)

# Build node_rows using this map
node_rows <- setNames(
  lapply(all_party_ids, function(pid) {
    rn <- party_to_rpart[as.character(pid)]
    if (is.na(rn)) return(integer(0))
    leaves_rpart <- rpart_terminals_of[[as.character(rn)]]
    party_leaves <- as.integer(rpart_to_party[as.character(leaves_rpart)])
    which(df$party_node %in% party_leaves)
  }),
  as.character(all_party_ids)
)

cat("\nFull-sample n per partykit node:\n")
print(sapply(node_rows, length))

# ── Stats helper ──────────────────────────────────────────────────────────────
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

cat("\nStat strings per node:\n"); print(node_stat_strings)

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

# ── Root stats from full sample ───────────────────────────────────────────────
root_x <- df$risk_score_12m[!is.na(df$risk_score_12m)]
root_n <- length(root_x); root_mean <- round(mean(root_x), 3)
root_se <- sd(root_x) / sqrt(root_n)
root_ci_lo <- round(root_mean - qt(0.975, df = root_n - 1) * root_se, 3)
root_ci_hi <- round(root_mean + qt(0.975, df = root_n - 1) * root_se, 3)
root_stat_str <- paste0("\u03bc = ", root_mean, "  [", root_ci_lo, ", ", root_ci_hi, "]\n",
                         "n = ", formatC(root_n, format = "d", big.mark = ","))

terminal_ids  <- nodeids(party_tree, terminal = TRUE)
inner_ids     <- setdiff(all_party_ids, terminal_ids)
nonroot_inner <- setdiff(inner_ids, 1L)
nonroot_all   <- setdiff(all_party_ids, 1L)

# ── Colors ────────────────────────────────────────────────────────────────────
node_bg <- "#003366"; root_bg <- "#F47920"
edge_fill <- "#E8F4FB"; edge_col <- "#0057A8"; text_col <- "#FFFFFF"

# ── Plot ──────────────────────────────────────────────────────────────────────
p <- ggparty(party_tree, terminal_space = 0.15,
             add_vars = list(
               node_stat   = function(data, node) node_stat_strings[as.character(node$id)],
               node_header = function(data, node) {
                 nid <- as.character(node$id)
                 if (node$id == 1L) "All Patients"
                 else if (node$id %in% terminal_ids) ""
                 else { lbl <- node_splitvar_label[nid]; if (is.na(lbl)) "" else lbl }
               }
             )) +
  geom_edge(color = "#00A3E0", linewidth = 0.7) +
  geom_edge_label(
    aes(label = ifelse(grepl(">=", breaks_label), "True",
                       ifelse(grepl("<", breaks_label), "False",
                              as.character(breaks_label)))),
    parse = FALSE, color = edge_col, size = 3.5, fontface = "bold",
    fill = edge_fill, label.padding = unit(0.25, "lines"),
    label.r = unit(0.12, "lines"), label.size = 0.3, shift = 0.28) +
  geom_edge_label(
    aes(label = node_stat_strings[as.character(id)]),
    ids = nonroot_all, parse = FALSE, color = edge_col, size = 2.8,
    fontface = "plain", fill = edge_fill,
    label.padding = unit(0.3, "lines"), label.r = unit(0.12, "lines"),
    label.size = 0.3, shift = 0.68) +
  geom_node_label(
    aes(label = paste0("All Patients\n", root_stat_str)),
    ids = 1L, color = text_col, fill = root_bg, size = 3.2, fontface = "bold",
    label.padding = unit(0.5, "lines"), label.r = unit(0.2, "lines"),
    label.col = root_bg) +
  geom_node_label(
    aes(label = ifelse(is.na(node_splitvar_label[as.character(id)]), "",
                       node_splitvar_label[as.character(id)])),
    ids = nonroot_inner, color = text_col, fill = node_bg, size = 3.5,
    fontface = "bold", label.padding = unit(0.4, "lines"),
    label.r = unit(0.2, "lines"), label.col = node_bg) +
  geom_node_label(
    aes(label = ""), ids = terminal_ids,
    color = "#F0F8FF", fill = "#F0F8FF", size = 0.1,
    label.padding = unit(0.05, "lines"), label.r = unit(0, "lines"),
    label.size = 0) +
  theme_void(base_family = "sans") +
  theme(
    plot.background = element_rect(fill = "#F0F8FF", color = NA),
    plot.title    = element_text(size = 18, face = "bold", color = "#003366",
                                 hjust = 0.5, margin = margin(b = 6)),
    plot.subtitle = element_text(size = 11, color = "#0057A8",
                                 hjust = 0.5, margin = margin(b = 16)),
    plot.caption  = element_text(size = 8, color = "#00A3E0", hjust = 1),
    plot.margin   = margin(20, 20, 20, 20)) +
  labs(
    title    = "Predictors of 12-Month MSK Risk Score",
    subtitle = paste0("Regression tree \u00b7 edge labels show full-sample mean (95% CI) and member count\n",
                      "Orange root = all ", formatC(root_n, format = "d", big.mark = ","),
                      " members \u00b7 navy nodes = split variable"),
    caption  = paste0("Tree fitted on 80% training split, pruned via 1-SE rule",
                      " \u00b7 statistics from full sample (n = ",
                      formatC(root_n, format = "d", big.mark = ","), ")"))

ggsave("decision_tree_msk_ggparty.png", plot = p,
       width = 16, height = 10, dpi = 300, bg = "#F0F8FF")
cat("Saved: decision_tree_msk_ggparty.png\n")
