# Load necessary libraries
library(rpart)        # Decision Trees
library(rpart.plot)   # Decision Tree visualization
library(class)        # K-Nearest Neighbors
library(gbm)          # Gradient Boosting
library(caret)        # For data splitting and evaluation

# Load the data
wdbc.data <- read.table(file = "~/Downloads/wdbc.data", sep = ",", header = FALSE)
names(wdbc.data) <- c("ID", "Diagnosis", paste0("Feature_", 1:30))
wdbc.data$Diagnosis <- as.factor(wdbc.data$Diagnosis)  # Convert to factor

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(wdbc.data$Diagnosis, p = 0.8, list = FALSE)
train <- wdbc.data[trainIndex, -1]  # Exclude ID column
test <- wdbc.data[-trainIndex, -1]




# --- Decision Tree ---
# Train Decision Tree with default parameters
dt_model <- rpart(Diagnosis ~ ., data = train, method = "class")

# Visualize the Decision Tree
rpart.plot(dt_model, main = "Decision Tree")

# Tune tree depth to analyze overfitting
dt_model_tuned <- rpart(Diagnosis ~ ., data = train, method = "class", 
                        control = rpart.control(maxdepth = 5))

# Predict and calculate accuracy
dt_preds_tuned <- predict(dt_model_tuned, test, type = "class")
dt_accuracy_tuned <- confusionMatrix(dt_preds_tuned, test$Diagnosis)$overall["Accuracy"]




# --- k-Nearest Neighbors ---
# Experiment with different k values
k_values <- seq(1, 15, 2)
knn_accuracies <- sapply(k_values, function(k) {
  knn_preds <- knn(train = train[, -1], test = test[, -1], cl = train$Diagnosis, k = k)
  confusionMatrix(knn_preds, test$Diagnosis)$overall["Accuracy"]
})

# Plot k-NN accuracy vs. k
plot(k_values, knn_accuracies, type = "b", 
     main = "k-NN Accuracy vs. k", xlab = "k", ylab = "Accuracy")

# Use the best k
best_k <- k_values[which.max(knn_accuracies)]
knn_preds_best <- knn(train = train[, -1], test = test[, -1], cl = train$Diagnosis, k = best_k)

# Perform k-NN for k = 15
k <- 15
knn_preds_k15 <- knn(train = train[, -1], test = test[, -1], cl = train$Diagnosis, k = k)

# Calculate accuracy for k = 15
knn_accuracy_k15 <- confusionMatrix(knn_preds_k15, test$Diagnosis)$overall["Accuracy"]

# Print the accuracy for k = 15
cat("k-NN Accuracy for k =", k, ":", knn_accuracy_k15, "\n")




# --- Gradient Boosting ---
# Convert Diagnosis to numeric
train$Diagnosis <- ifelse(train$Diagnosis == "M", 1, 0)  # Malignant = 1, Benign = 0
test$Diagnosis_numeric <- ifelse(test$Diagnosis == "M", 1, 0)  # Convert for comparison

# Train Gradient Boosting model
gbm_model <- gbm(Diagnosis ~ ., 
                 data = train, 
                 distribution = "bernoulli", 
                 n.trees = 100, 
                 interaction.depth = 3, 
                 shrinkage = 0.01, 
                 cv.folds = 5, 
                 verbose = FALSE)

# Plot the Bernoulli deviance
gbm.perf(gbm_model, method = "cv")

# Predictions
gbm_preds <- predict(gbm_model, test[setdiff(names(test), c("ID", "Diagnosis_numeric"))], 
                     n.trees = 100, type = "response")
gbm_preds_class <- ifelse(gbm_preds > 0.5, 1, 0)  # Convert probabilities to binary

# Accuracy
gbm_accuracy <- confusionMatrix(as.factor(gbm_preds_class), as.factor(test$Diagnosis_numeric))$overall["Accuracy"]

# Detailed confusion matrix
conf_matrix <- confusionMatrix(as.factor(gbm_preds_class), as.factor(test$Diagnosis_numeric))
print(conf_matrix)

# --- Output Results ---
cat("Decision Tree Accuracy (Tuned):", dt_accuracy_tuned, "\n")
cat("k-NN Best Accuracy (k =", best_k, "):", max(knn_accuracies), "\n")
cat("Gradient Boosting Accuracy:", gbm_accuracy, "\n")
