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