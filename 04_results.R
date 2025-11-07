# Import libraries and source utility functions ----
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, arrow, stargazer, pscl, tidyr, caret, ggrepel)
`%ni%` <- Negate(`%in%`)
source('utils.R')

# Load data ----
data <- read_parquet("data/processed_data/2025-06-01T19-09-22.parquet") %>% 
  dplyr::select(!c("store_pickup", "seller_nickname")) %>% 
  mutate(
    flex_and_full = flex_shipping * fullfilment
  )

# Table 1 - Descriptive statistics ----
data_summary <- data %>% 
  dplyr::select(!c(
    "product_id", "seller_id", "premium_listing_pct",
    "official_store_pct", "free_shipping_pct", "fullfilment_pct",
    "flex_shipping_pct", "seller_level_mean", "power_seller_mean",
    "seller_transactions_total_mean"
  )) 

pct_features <- c(
  "premium_listing", "official_store", "meli_direct_seller", 
  "free_shipping", "fullfilment", "cross_docking",
  "flex_shipping", "flex_and_full", "southeast_seller", 
  "buy_box_winner", "pct_price_diff"
)

summary <- as.data.frame(apply(data_summary, 2, summary)) %>% 
  mutate_at(
    vars(all_of(pct_features)),
    ~ .x * 100
  ) 

summary[setdiff(rownames(summary), "Mean"), setdiff(pct_features, "pct_price_diff")] <- NA

summary <- summary %>% 
  mutate(
    across(everything(), ~ round(.x, 2))
  ) %>% 
  t() %>% 
  as.data.frame() %>% 
  mutate(across(everything(), as.character)) %>%  # Convert everything to character
  mutate(across(everything(), ~ replace_na(.x, "-")))

rownames(summary) <- sapply(rownames(summary), function(x) ifelse(x %in% pct_features, paste0(x, " (%)"), x))

rownames(summary) <- sapply(rownames(summary), format_names)

rownames(summary) <- sapply(rownames(summary), function(x) ifelse(x == "Pct Price Diff (%)", "Price Diff (%)", x))

stargazer(summary, type = "latex")

stargazer(summary, 
          type = "latex", 
          summary = FALSE,  # Disable summary statistics
          rownames = TRUE)

# Table 2 Logistic regression ----
regressors_variations <- list(
  c(
    "price_diff", "pct_price_diff", "premium_listing", "official_store",
    "meli_direct_seller", "free_shipping", "fullfilment", "cross_docking",
    "flex_shipping", "flex_and_full", "southeast_seller", "seller_level", "power_seller",
    "seller_transactions_total", "n_competitors", "premium_listing_pct",
    "official_store_pct", "free_shipping_pct", "fullfilment_pct",
    "flex_shipping_pct", "seller_level_mean", "power_seller_mean",
    "seller_transactions_total_mean"
  ),
  c(
    "price_diff", "pct_price_diff", "premium_listing", "official_store",
    "meli_direct_seller", "free_shipping", "fullfilment", "cross_docking",
    "flex_shipping", "flex_and_full", "southeast_seller", "seller_level", 
    "power_seller", "seller_transactions_total"
  ),
  c(
    "price_diff", "pct_price_diff"
  )
)

logistic_models <- list()
for(i in seq_along(regressors_variations)){
  model_df <- data %>% 
    dplyr::select(
      buy_box_winner,
      regressors_variations[[i]]
    )
  
  logit_model <- glm(
    buy_box_winner ~ ., 
    data = model_df,
    family = binomial(link = "logit")
  )
  
  logistic_models[[i]] <- logit_model
}

# Anova test for nested models
anova1_2 <- anova(logistic_models[[2]], logistic_models[[1]], test = "Chisq")

print(paste(
  "Chi-squared test between models 1 and 2 is", anova1_2[2, "Deviance"], 
  "with p-value", anova1_2[2, "Pr(>Chi)"],
  "and degrees of freedom", anova1_2[2, "Df"]
))

anova2_3 <- anova(logistic_models[[3]], logistic_models[[2]], test = "Chisq")

print(paste(
  "Chi-squared test between models 2 and 3 is", anova2_3[2, "Deviance"], 
  "with p-value", anova2_3[2, "Pr(>Chi)"],
  "and degrees of freedom", anova2_3[2, "Df"]
))

# McFadden's pseudo R-squared
for (i in seq_along(logistic_models)){
  mcfadden_pr2 <- pscl::pR2(logistic_models[[i]])
  print(paste("McFadden's pseudo R-squared for model", i, "is", mcfadden_pr2["McFadden"]))
}

# Dummy probabilities based on coefficients 
dummy_variables <- c(
  "premium_listing", "official_store",
  "meli_direct_seller", "free_shipping", 
  "fullfilment", "cross_docking",
  "flex_shipping", "flex_and_full", "southeast_seller"
)

dummy_coefficients <- logistic_models[[1]]$coefficients[dummy_variables]

sapply(dummy_coefficients, function(x) (exp(x) - 1) * 100)

# LaTeX table
stargazer(
  logistic_models,
  title="Logisitic Regression", align=TRUE, 
  covariate.labels = sapply(regressors_variations[[1]], format_names),
  dep.var.labels = "Buy Box Winner",
  no.space = TRUE, single.row=TRUE,
  column.sep.width="1pt"
)

# Table 3 - Prediction Results ----
linear_models <- c("LogisticRegression")
ensemble_models <- c("RandomForestClassifier", "LGBMClassifier", "CatBoostClassifier", "XGBClassifier")
deep_learning_models <- c("TabNetClassifier")
dummy_models <- c("LowestPriceDummy", "LowestPriceFullfilmentDummy")

predictions <- read_parquet("data/predictions/2025-06-01T19-26-44.parquet")

model_pred <- predictions %>% 
  dplyr::filter(model_name == "LGBMClassifier")

actual <- model_pred$buy_box_winner
predicted <- model_pred$max_prob_flag

prediction_results_data <- predictions %>% 
  group_by(model_name) %>%
  summarise(
    precision = precision_recall_f1(buy_box_winner, max_prob_flag)[1],
    recall = precision_recall_f1(buy_box_winner, max_prob_flag)[2],
    f1_score = precision_recall_f1(buy_box_winner, max_prob_flag)[3],
    mcc = matthews_correlation_coefficient(buy_box_winner, max_prob_flag),
    running_time = mean(execution_time)
  ) %>% 
  #Format table to export
  mutate(model_name = recode(
    model_name,
    "CatBoostClassifier" = "CatBoost",
    "LGBMClassifier" = "Light GBM",
    "LogisticRegression" = "Logistic Reg.",
    "LowestPriceDummy" = "Dummy Price",
    "LowestPriceFullfilmentDummy" = "Dummy Price-Full",
    "RandomForestClassifier" = "Random Forest",
    "TabNetClassifier" = "TabNet",
    "XGBClassifier" = "XGBoost"
  )) %>% 
  mutate_at(
    vars(precision, recall, f1_score, mcc),
    ~ round(.x*100, 2)
  ) %>%
  mutate_at(
    vars(running_time),
    ~ round(.x, 2)
  ) %>%
  arrange(
    desc(precision)
  ) %>%
  rename_with(
    ~ format_names(.x)
  ) %>% 
  rename(`MCC` = `Mcc`, `Run. Time` = `Running Time`) %>% 
  column_to_rownames(var = "Model Name")

stargazer(prediction_results_data, 
          type = "latex", 
          summary = FALSE,  # Disable summary statistics
          rownames = TRUE,
          title = "Prediction Results",
          digits=2)

# AUC-PR
probs <- lgbm_results$y_pred_prob

fg <- probs[lgbm_results$buy_box_winner == 1]
bg <- probs[lgbm_results$buy_box_winner == 0]

roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)

pr.curve(lgbm_results$buy_box_winner, lgbm_results$max_prob_flag, curve = TRUE)


figure3 <- ggplot(prediction_results_data, aes(x = running_time, y = precision)) +
  geom_point(aes(color = model_group)) + #size = running_time,
  geom_label_repel(label = prediction_results_data$model_name,  size=3.5) +
  scale_size_continuous(range = c(1, 15)) +
  scale_y_continuous(labels = scales::percent) +
  theme_classic() +
  xlab("Precision") + ylab("Recall") +
  labs(color = "Model", size = "Exe. Time") + 
  theme(axis.title.x = element_text(vjust=-0.5),
        axis.title.y = element_text(vjust=1.5))

rm(running_time, tercile_returns_regres, out_of_sample_r2, plot3_data)