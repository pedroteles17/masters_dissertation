# Import libraries and source utility functions ----
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, arrow, stargazer, pscl, tidyr)
`%ni%` <- Negate(`%in%`)
source('utils.R')

# Load data ----
data <- read_parquet("data/processed_data.parquet") %>% 
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
