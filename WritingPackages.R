####################
# Writing Packages #
####################

###### The R Package Structure

### The structure of an R package

# Use the create function to set up your first package
create("datasummary")

# Take a look at the files and folders in your package
dir("datasummary")


### Writing a simple function
# Create numeric_summary() function
numeric_summary <- function(x, na.rm) {
  
  # Include an error if x is not numeric
  if(!is.numeric(x)){
    return("Data must be numeric")
    
    stop()
  }
  
  # Create data frame
  data.frame( min = min(x, na.rm = na.rm),
              median = median(x, na.rm = na.rm),
              sd = sd(x, na.rm = na.rm),
              max = max(x, na.rm = na.rm))
  
}

# Test numeric_summary() function
numeric_summary(airquality$Ozone, TRUE)

### Including functions in a package

# What is in the R directory before adding a function?
dir("datasummary/R")

# Use the dump() function to write the numeric_summary function
dump("numeric_summary", file = "datasummary/R/numeric_summary.R")

# Verify that the file is in the correct directory
dir("datasummary/R")

### The use_* functions

# What is in the package at the moment?
dir("datasummary")

# Add the weather data
use_data(weather, pkg = "datasummary", overwrite = TRUE)

# Add a vignette called "Generating Summaries with Data Summary"
use_vignette("Generating_Summaries_with_Data_Summary", pkg = "datasummary")

# What directories do you now have in your package now?
dir("datasummary")

### Best practice for structuring code
data_summary <- function(x, na.rm = TRUE){
  
  num_data <- select_if(x, .predicate = is.numeric) 
  
  map_df(num_data, .f = numeric_summary, na.rm = TRUE, .id = "ID")
  
}

# Write the function to the R directory
dump("data_summary", file = "datasummary/R/data_summary.R")

### A simple function header

# Add a title and description
#'Numeric Summaries
#'
#'Summarises numeric data and returns a data frame containing the minimum value, median, standard deviation, and maximum value.
#'
numeric_summary <- function(x, na.rm){
  
  if(!is.numeric(x)){
    stop("Data must be numeric")
  }
  
  data.frame( min = min(x, na.rm = na.rm),
              median = median(x, na.rm = na.rm),
              sd = sd(x, na.rm = na.rm),
              max = max(x, na.rm = na.rm))
}


### Documenting function arguments

#' Numeric Summaries
#'
#' Summarises numeric data and returns a data frame containing the minimum value, median, standard deviation, and maximum value.
#'
# Add appropriate tag and details to document the first argument
#' @param x a numeric vector containing the values to summarize.
numeric_summary <- function(x, na.rm){
  
  if(!is.numeric(x)){
    stop("data must be numeric")
  }
  
  data.frame( min = min(x, na.rm = na.rm),
              median = median(x, na.rm = na.rm),
              sd = sd(x, na.rm = na.rm),
              max = max(x, na.rm = na.rm))
}

### Importing other packages

