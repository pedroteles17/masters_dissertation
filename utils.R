
format_names <- function(x){
  x <- gsub("_", " ", x)
  x <- tolower(x)
  x <- stringr::str_to_title(x)
  return(x)
}

precision_recall_f1 <- function(actual, predicted){
  tp <- sum(predicted == 1 & actual == 1)  # True Positives
  fp <- sum(predicted == 1 & actual == 0)  # False Positives
  fn <- sum(predicted == 0 & actual == 1)  # False Negatives
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  return(c(precision, recall, f1_score))
}

matthews_correlation_coefficient <- function(actual, predicted){
  tp <- as.numeric(sum(predicted == 1 & actual == 1))  # True Positives
  tn <- as.numeric(sum(predicted == 0 & actual == 0))  # True Negatives
  fp <- as.numeric(sum(predicted == 1 & actual == 0))  # False Positives
  fn <- as.numeric(sum(predicted == 0 & actual == 1))  # False Negatives
  
  mcc <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  return(mcc)
}