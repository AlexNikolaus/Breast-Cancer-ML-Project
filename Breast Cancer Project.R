library(tidyverse)
library(caret)

#Set filepath and read data
filepath = file.path("D:/R Projects/Infarction project")
Data <- read.csv("data.csv")

#Create separate matrix for means
Matrix_means <- Data[,2:12]
rownames(Matrix_means) <- Data[,1]

#Set seed and check for any NAs
set.seed(4733)
anyNA(Matrix_means)

#Pre processing
Matrix_preprocess <- preProcess(Matrix_means, method = c("center","nzv"))
data_processed <- predict(Matrix_preprocess, Matrix_means)

#Create data partition 75-25%
index <- createDataPartition(data_processed$diagnosis, p=0.75, list = FALSE)
trainSet <- data_processed[index,]
testSet <- data_processed[-index,]

#Set up 10-fold cross-validation
trainControl <- trainControl(method = "cv", number = 10, savePredictions = 'final', classProbs = T)
#No need to define predictor or outcome names, as I'll only use the means to predict diagnostic

#Train the k-NN model
knnfit <- train( x = trainSet[,-1],
                 y = trainSet[,1],
                 method = "knn",
                 trControl = trainControl,
                 tuneGrid = data.frame(k = 1:10))

#Predictions and accuracy on the training set using k-NN
trainpred_knn <- predict(knnfit, trainSet[,-1])
confusionMatrix(trainpred_knn, as.factor(trainSet[,1]))

#Train the random forest
rffit <- train( x = trainSet[,-1],
                y = trainSet[,1],
                method = "ranger",
                trControl = trainControl,
                tuneGrid = data.frame(mtry = 1:10,
                                      splitrule = "gini",
                                      min.node.size = 1))

#Predictions and accuracy on the training set using Random forests
trainpred_rf <- predict(rffit, trainSet[,-1])
confusionMatrix(trainpred_rf, as.factor(trainSet[,1]))


#Train the xgboost
xgboost_fit <- train( x = trainSet[,-1],
                      y = trainSet[,1],
                      method = "xgbLinear",
                      trControl = trainControl,
                      tuneGrid = data.frame(nrounds = 10,
                                            lambda = 0.5,
                                            alpha = 0.5,
                                            eta = c(0.3:0.7)))

#Prediction and accuracy on the training set using xgboosting
trainpred_xgboost <- predict(xgboost_fit, trainSet[,-1])
confusionMatrix(trainpred_xgboost, as.factor(trainSet[,1]))

#Use ensemble learning and voting for the final model
# predict test with k-NN model
knnPred=as.character(predict(knnfit,testSet[,-1]))
# predict test with Random forest model
rfPred=as.character(predict(rffit,testSet[,-1]))
# predict test with xgboost model
xgbPred=as.character(predict(xgboost_fit,testSet[,-1]))

# voting for class labels
# code finds the most frequent class label per row
votingPred=apply(cbind(knnPred,rfPred,xgbPred),1,
                 function(x) names(which.max(table(x))))

#Check accuracy
confusionMatrix(data=as.factor(testSet[,1]),
                reference=as.factor(votingPred))
