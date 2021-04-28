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
