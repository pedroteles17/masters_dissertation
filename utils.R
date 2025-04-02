
format_names <- function(x){
  x <- gsub("_", " ", x)
  x <- tolower(x)
  x <- stringr::str_to_title(x)
  return(x)
}