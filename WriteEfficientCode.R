##########################
# Write Efficient R Code #
##########################

###### Benchmarking

### R version

# Print the R version details using version
version

# Assign the variable major to the major component
major <- version$major

# Assign the variable minor to the minor component
minor <- version$minor

### Elapsed time

# How long does it take to read movies from CSV?
system.time(read.csv("movies.csv"))

# How long does it take to read movies from RDS?
system.time(readRDS("movies.rds"))

### Comparing read times of CSV and RDS files

# Load the microbenchmark package
library(microbenchmark)

# Compare the two functions
compare <- microbenchmark(read.csv("movies.csv"), 
                          readRDS("movies.rds"), 
                          times = 10)

# Print compare
compare

### Hardware

# Load the benchmarkme package
library(benchmarkme)

# Assign the variable ram to the amount of RAM on this machine
ram <- get_ram()
ram

# Assign the variable cpu to the cpu specs
cpu <- get_cpu()
cpu

### Benchmark machine

# Load the package
library("benchmarkme")

# Run the io benchmark
res <- benchmark_io(runs = 1, size = 5)

# Plot the results
plot(res)

###### Fine Tuning: Efficient Base R

### Timings - growing a vector

# Use <- with system.time() to store the result as res_grow
system.time(res_grow <- growing(n))


### Timings - pre-allocation

# Use <- with system.time() to store the result as res_allocate
n <- 30000
system.time(res_allocate <- pre_allocate(n))

### Vectorized code: multiplication

# Store your answer as x2_imp
x2_imp <- x * x

### Vectorized code: calculating a log-sum

# Initial code
n <- 100
total <- 0
x <- runif(n)
for(i in 1:n) 
  total <- total + log(x[i])

# Rewrite in a single line. Store the result in log_sum
log_sum <- sum(log(x[1:100]))




