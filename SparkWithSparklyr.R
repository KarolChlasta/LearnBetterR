#######################
# Spark with sparklyr #
#######################

install.packages("sparklyr")

###### Starting To Use Spark With dplyr Syntax

### The connect-work-disconnect pattern
# Load sparklyr
library(sparklyr)

# Connect to your Spark cluster
spark_conn <- spark_connect(master = 'local')

# Print the version of Spark
spark_version(sc = spark_conn)

# Disconnect from Spark
spark_disconnect(sc = spark_conn)

### Copying data into Spark
# Load dplyr
library(dplyr)

# Explore track_metadata structure
str(track_metadata, overwrite = TRUE)

# Connect to your Spark cluster
spark_conn <- spark_connect("local")

# Copy track_metadata to Spark
track_metadata_tbl <- copy_to(spark_conn, track_metadata, overwrite = TRUE)

# List the data frames available in Spark
src_tbls(spark_conn)

# Disconnect from Spark
spark_disconnect(spark_conn)

### Big data, tiny tibble
# Link to the track_metadata table in Spark
track_metadata_tbl <- tbl(spark_conn, "track_metadata")

# See how big the dataset is
dim(track_metadata_tbl)

# See how small the tibble is
object_size(track_metadata_tbl)

### Exploring the structure of tibbles
# Print 5 rows, all columns
print(track_metadata_tbl, n=5, width = Inf)

# Examine structure of tibble
str(track_metadata_tbl)

# Examine structure of data
glimpse(track_metadata_tbl)

### Selecting columns
# track_metadata_tbl has been pre-defined
track_metadata_tbl

# Manipulate the track metadata
track_metadata_tbl %>%
  # Select columns
  select(artist_name, release, title, year)

# Try to select columns using [ ]
tryCatch({
  # Selection code here
  track_metadata_tbl[, c("artist_name", "release", "title", "year")]
},
error = print
)

### Filtering rows
# track_metadata_tbl has been pre-defined
glimpse(track_metadata_tbl)

# Manipulate the track metadata
track_metadata_tbl %>%
  # Select columns
  select(artist_name, release, title, year) %>%
  # Filter rows
  filter(year >= 1960, year < 1970)

### Arranging rows
# track_metadata_tbl has been pre-defined
track_metadata_tbl

# Manipulate the track metadata
track_metadata_tbl %>%
  # Select columns
  select(artist_name, release, title, year) %>%
  # Filter rows
  filter(year >= 1960, year < 1970) %>%
  # Arrange rows
  arrange(artist_name, desc(year), title)

### Mutating columns
# track_metadata_tbl has been pre-defined
track_metadata_tbl

# Manipulate the track metadata
track_metadata_tbl %>%
  # Select columns
  select(title, duration) %>%
  # Mutate columns
  mutate( duration_minutes = duration / 60)

### Summarizing columns
# track_metadata_tbl has been pre-defined
track_metadata_tbl

# Manipulate the track metadata
track_metadata_tbl %>%
  # Select columns
  select(title, duration) %>%
  # Mutate columns
  mutate(duration_minutes = duration / 60) %>%
  # Summarize columns
  summarize(mean_duration_minutes = mean(duration_minutes))

######
