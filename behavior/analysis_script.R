library(dplyr)

# ---- Read the data ----
dat <- read.csv(
  "Behavior_data.csv",
  na.strings = c("NA", ""),
  stringsAsFactors = FALSE,
  strip.white = TRUE
)

# ---- Select participants in the deep and shallow groups ----
dat_group <- dat %>%
  mutate(group = trimws(group)) %>%
  filter(group %in% c("deep", "shallow"))

# ---- Calculate the mean of each numeric column by group ----
# Missing values are excluded separately for each column.
group_means <- dat_group %>%
  group_by(group) %>%
  summarise(
    n = n(),
    across(
      where(is.numeric),
      ~ if (all(is.na(.x))) NA_real_ else mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

print(group_means, width = Inf)

# ---- Calculate descriptive statistics for TrialRTB ----
trialrtb_summary <- dat_group %>%
  group_by(group) %>%
  summarise(
    n_valid = sum(!is.na(TrialRTB)),
    n_missing = sum(is.na(TrialRTB)),
    Mean = mean(TrialRTB, na.rm = TRUE),
    SD = sd(TrialRTB, na.rm = TRUE),
    Median = median(TrialRTB, na.rm = TRUE),
    Q1 = quantile(TrialRTB, 0.25, na.rm = TRUE),
    Q3 = quantile(TrialRTB, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

print(trialrtb_summary, width = Inf)

# ---- Extract non-missing TrialRTB values for each group ----
deep_values <- dat_group %>%
  filter(group == "deep", !is.na(TrialRTB)) %>%
  pull(TrialRTB)

shallow_values <- dat_group %>%
  filter(group == "shallow", !is.na(TrialRTB)) %>%
  pull(TrialRTB)

# ---- Perform a two-sided Wilcoxon rank-sum test ----
# Use a normal approximation with continuity correction.
# The W statistic is reported with deep as the first group.
wilcox_result <- wilcox.test(
  x = deep_values,
  y = shallow_values,
  alternative = "two.sided",
  paired = FALSE,
  exact = FALSE,
  correct = TRUE
)

print(wilcox_result)
