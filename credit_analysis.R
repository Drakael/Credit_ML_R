library(gbm)
library(onehot)
library(caret)
getwd()
setwd('C:/Users/Drakael/Downloads/DATA IA/Cours I.A/36 - R')

header = c("Comptes", "Duree_credit", "Historique_credit", "Objet_credit", "Montant_credit", "Epargne", "Anciennete_emploi", "taux_effort", "Situation_familiale", "Garanties", "Anciennete_domicile", "Biens", "Age", "Autres_credits", "Statut_domicile", "Nb_credits", "Type_emploi", "Nb_pers_charge", "Telephone", "Etranger", "Cible")

credit_file <- read.table('credit.txt', header=FALSE, col.names = header)
credit_file
df <- data.frame(credit_file, sep=' ')
# View(df)

# summary(df)

smp_size <- floor(0.8 * nrow(df))

# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

smp_size <- floor(0.8 * nrow(train))
valid_ind <- sample(seq_len(nrow(train)), size = smp_size)
train <- train[valid_ind, ]
valid <- train[-valid_ind, ]

print(nrow(train))
print(nrow(valid))
print(nrow(test))


header_X = c("Comptes", "Duree_credit", "Historique_credit", "Objet_credit", "Montant_credit", "Epargne", "Anciennete_emploi", "taux_effort", "Situation_familiale", "Garanties", "Anciennete_domicile", "Biens", "Age", "Autres_credits", "Statut_domicile", "Nb_credits", "Type_emploi", "Nb_pers_charge", "Telephone", "Etranger")
(X_train <- train[header_X])
(y_train <- train["Cible"])
(X_valid <- valid[header_X])
(y_valid <- valid["Cible"])
(X_test <- test[header_X])
(y_test <- test["Cible"])

train_onehot <- onehot(X_train, stringsAsFactors=TRUE)
valid_onehot <- onehot(X_valid, stringsAsFactors=TRUE)
test_onehot <- onehot(X_test, stringsAsFactors=TRUE)

X_train_onehot <- predict(train_onehot, X_train)
X_valid_onehot <- predict(valid_onehot, X_valid)
X_test_onehot <- predict(test_onehot, X_test)

summary(X_train_onehot)
summary(X_valid_onehot)
summary(X_test_onehot)

print(nrow(X_train_onehot))
print(nrow(X_valid_onehot))
print(nrow(X_test_onehot))



LogLossBinary = function(actual, predicted, eps = 1e-15) {  
  predicted = pmin(pmax(predicted, eps), 1-eps)  
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}



gbmModel = gbm(formula = Cible ~ .,
               distribution = "gaussian",
               data = train,
               verbose=TRUE,
               interaction.depth=4,
               n.trees = 1500,
               shrinkage = .01,
               cv.folds=10,
               n.minobsinnode = 20)

gbmTrainPredictions = predict(object = gbmModel,
                              newdata = test,
                              n.trees = 1500,
                              type = "response")

predicted_classes <- gbmTrainPredictions

predicted_classes[predicted_classes<=1.5] <- 1
predicted_classes[predicted_classes>1.5] <- 2

print(gbmTrainPredictions)
print(predicted_classes)

head(data.frame("Actual" = test$Cible, 
                "PredictedProbability" = predicted_classes))

LogLossBinary(test$Cible, predicted_classes)

nb_errors <- sum(abs(test$Cible - predicted_classes))
accuracy <- (length(test$Cible) - nb_errors) / length(test$Cible)
print(nb_errors)
print(length(test$Cible))
print(accuracy)

par(mar=c(3,14,1,1))
summary(gbmModel, las=2)
