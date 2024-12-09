library(poLCA)
# Set seed for reproducibility
set.seed(10)

# Load data
data <- read.csv("data_parsed.csv", stringsAsFactors = FALSE)

# Convert Python set from csv file to R vector
data$Tags <- lapply(data$Tags, function(tag_set) {
  clean_string <- gsub("[{}']", "", tag_set)
  trimws(strsplit(clean_string, ",")[[1]])
})

# Generate a unique list of all tags
all_tags <- unique(unlist(data$Tags))

# Generate a one-hot matrix
tag_matrix <- sapply(all_tags, function(tag) {
  sapply(data$Tags, function(tags) as.integer(tag %in% tags))
})

# Convert matrix to a dataframe and increment by 1 for LCA analysis
tag_matrix <- as.data.frame(tag_matrix)
tag_matrix <- as.data.frame(lapply(tag_matrix, as.integer)) + 1

# Create formula for poLCA
formula <- as.formula(paste("cbind(", paste(names(tag_matrix), 
                                            collapse = ", "), ") ~ 1"))

# Fit LCA model with 8 classes
lca_model <- poLCA(formula, 
                   data = tag_matrix, 
                   nclass = 8, 
                   nrep = 3, 
                   maxiter = 5000)

# Extract conditional probabilities from the model
conditional_probs <- lca_model$probs

# Calculate probabilities of a tag belonging to each class
tag_class_probs <- sapply(conditional_probs, 
                          function(tag_probs) tag_probs[, "Pr(2)"])

# Assign each tag to its most probable class
tag_to_class <- apply(tag_class_probs, 2, which.max)

# Group tags by class
tags_by_class <- split(names(tag_to_class), tag_to_class)

# Output: Tags grouped by class
tags_by_class
