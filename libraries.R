#### Libraries needed ####

library(ggplot2)
library(gridExtra)
library(keras)
library(reticulate)
require(keras)

use_condaenv("base")
py_discover_config('keras')