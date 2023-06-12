library(data.table)
library(xgboost)
library(caret)
library(Rcpp)
library(lhs)

library(DiceKriging)
library(DiceOptim)
library(DiceView)



# git hub data 
# DT = fread("https://raw.githubusercontent.com/bhimmetoglu/RoboBohr/master/data/outComes.csv")

outcomes <- fread(
  "https://raw.githubusercontent.com/bhimmetoglu/RoboBohr/master/data/out.dat.0_16274", skip = 2, header = FALSE, colClasses = c("integer", "numeric"))
colnames(outcomes) <- c("Id", "E")


AeSub <- fread("https://raw.githubusercontent.com/bhimmetoglu/RoboBohr/master/data/AEsub.out", header = FALSE, colClasses = c("integer","numeric"))
colnames(AeSub) <- c("Id", "Esub")


outcomesAe <- merge(outcomes, AeSub, by = "Id")
outcomesAe[,Eat := E-Esub]
outcomesAe[,E:=NULL]; outcomesAe[,Esub:=NULL] # Remove unnescessary columns
rm(outcomes,AeSub)

CoulombLambda <- fread("https://raw.githubusercontent.com/bhimmetoglu/RoboBohr/master/data/coulombL.csv",
                       header = FALSE)

# Assign column names
nam <- paste0('px', 1:(ncol(CoulombLambda)-1))
nam <- c("Id", nam)
colnames(CoulombLambda) <- nam
CoulombLambda[,Id:=as.integer(Id)] # Make Id variable integer

# Match with Id's so that there is no mistmatch in order
combined <- merge(CoulombLambda, outcomesAe, by="Id")
rm(outcomesAe,CoulombLambda)

# Remove NA's (calculations that failed to converge)
l.complete <- complete.cases(combined$Eat)
combined <- combined[l.complete,]

# Store atomization energies in a vector Y and remove unnecessary columns from combined
Y <- combined$Eat
combined[,Eat:=NULL] # No need for Eat 
combined[,Id:=NULL] # No need for Id


dim(combined)
hist(Y)

set.seed(101)
inTrain <- sample(1:dim(combined)[1],
                  size = floor(0.7*dim(combined)[1]),
                  replace = FALSE)
train.Y <- Y[inTrain]; test.Y <- Y[-inTrain]
train.X <- combined[inTrain,]; test.X <- combined[-inTrain,]  

dtrain.X <- xgb.DMatrix(as.matrix(train.X), label = train.Y)
dtest.X <- xgb.DMatrix(as.matrix(test.X), label = test.Y)

param <- list(booster="gbtree",
              eval_metric="rmse",
              eta=0.0156,
              colsample_bytree = 0.4,
              max_depth = 8,
              min_child_weight = 10,
              gamma = 0.0,
              lambda = 1.0,
              subsample = 0.8)

xgb.model <- xgb.train(data=dtrain.X,
                       params = param,
                       nround = 600)

system.time({
mod.cv <- xgb.cv(data = dtrain.X,
                 params = param,
                 nfold = 5, nrounds = 600)
})

# user  system elapsed 
# 118.14   12.73   22.89 

Rydberg_to_kcal_mol = (3506.37)/11.17897
mod.cv$evaluation_log$test_rmse_mean[600]*Rydberg_to_kcal_mol


# Predict
pred <- predict(xgb.model, newdata = dtest.X)

# RMSE
sqrt(mean((pred - test.Y)^2)) 

# RMSE in kcal/mol SCALE 
sqrt(mean((pred - test.Y)^2)) *(3506.37)/11.17897

hist(pred - test.Y, 50, xlim =c(-1,1))

# useless stuff for fun 
qqnorm(pred-test.Y)
qqline(pred-test.Y)





xgb_grid <- expand.grid(eta = 2^seq(-6,-4),
                        colsample_bytree = c(0.2,0.4,0.6),
                        max_depth = c(2,6,8,16),
                        min_child_weight = c(2,6,8,10),
                        gamma = c(0,1e-4,0.001,0.01))


set.seed(1919) 

final_design <- maximinLHS(25, 6,
                 method="iterative",
                 eps=0.00005, maxIter=500,
                 optimize.on="grid")

final_design




# function for computer sim
inversenormalize = function(y , min_x , max_x){
  X = y * (max_x - min_x) + min_x
  return(X)
}

# computer simulation function 
computer_simulation_2 = function( X ){
  
  
  X = ifelse(X >1 , 1 , X )
  learn_rate  = X[1]
  max_depth   = X[2]
  min_node    = X[3]
  col_sample  = X[4]
  
  gamma       = X[5]
  #boost_round = X[6]
  print(paste0("Learn Rate", learn_rate))
  print(paste0("max depth", max_depth))
  print(paste0("min node", min_node))
  print(paste0("col sample", col_sample))
  print(paste0("gamma", gamma))
  #print(paste0("boost_round", boost_round))
  
  learn_rate = inversenormalize(learn_rate, 0.015625, 0.062500)
  max_depth  = ceiling( qunif(max_depth, 2 , 16 ) )
  min_node   = ceiling( qunif(min_node, 2 , 10))
  col_sample = inversenormalize(col_sample, 0.2, 0.6 )
  
  gamma     = inversenormalize(gamma , 0, 0.1 )
  
  #boost_round = ceiling(qunif(boost_round , 400, 800))
  
  

  
  # parameter set up 
  params <- list( objective = 'reg:squarederror',
                  booster = 'gbtree',
                  nthread = 12,
                  min_child_weight = min_node,
                  max_depth = max_depth, 
                  learning_rate = learn_rate,
                  colsample_bytree = col_sample ,
                  gamma = gamma,
                  subsample =0.8)
  
  
  set.seed(100)
  
  model2 = xgb.cv(params = params,
                  data = dtrain.X , 
                  nrounds = 600,
                  nfold = 5,
                  verbose = 0)
  
  performance = model2$evaluation_log$test_rmse_mean[600]
  
  # its RMSE so square it 
  results = performance
  
  # pause execution for a few
  # minutes to prevent overheating of GPU
  gc( reset = TRUE )
  #Sys.sleep(120L)
  print(results)
  return(results)
  
}

# time it / ensure it works 
system.time({
  computer_simulation_2(as.numeric(final_design[1, 1:6]))
})



results_2 = rep(0, 25)
system.time({
  for(i in 1:25){
    tmp = computer_simulation_2(as.numeric(final_design[i, 1:6]))
    results_2[i] = tmp 
    gc(reset = TRUE )
  }
})

full_results = data.frame(results = results_2,
                          learn_rate = final_design[,1],
                          max_depths = final_design[,2],
                          min_node   = final_design[,3],
                          col_samples = final_design[,4],
                          gamma       = final_design[,5],
                          boost_round = rep(400,25))



path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/energy_state_1.csv"
fwrite(full_results,path.out)

full_results$boost_round <- NULL
library(DiceKriging)
library(DiceOptim)
library(DiceView)
full_results$results <- NULL
model = km(formula = ~ ., 
           design = full_results, 
           response = results_2)

# need to make the plot window huge
DiceView::sectionview(model , 
                      center = rep(0.5,5), 
                      col_points = "black",
                      bg_blend = 0 ,
                      ylim = c(0.1, .3))

# select best parameter as init

res.nsteps <- EGO.nsteps(model = model ,
                         fun = computer_simulation_2,
                         nsteps = 15,
                         lower = rep(0, 5),
                         upper = rep(1, 5),
                         parinit = as.numeric(full_results[10,1:5]),
                         control = list(pop.size = 500,
                                        max.generations = 100,
                                        wait.generations = 25,
                                        BFGSburnin = 100))

res.nsteps$value*Rydberg_to_kcal_mol
min(res.nsteps$value*Rydberg_to_kcal_mol)
min(results_2)
which.min(res.nsteps$value)

# see results on test set
res.nsteps$par[4,]


param <- list(booster="gbtree",
              eval_metric="rmse",
              eta=inversenormalize(res.nsteps$par[12,1], 0.015625, 0.062500),
              colsample_bytree =inversenormalize(res.nsteps$par[12,4], 0.2, 0.6 ),
              max_depth = ceiling( qunif(res.nsteps$par[12,2], 2 , 16 ) ),
              min_child_weight =  ceiling( qunif(res.nsteps$par[12,3], 2 , 10)),
              gamma = inversenormalize(res.nsteps$par[12,5] , 0, 0.1 ),
              lambda = 1.0,
              subsample = 0.8)

xgb.model <- xgb.train(data=dtrain.X,
                       params = param,
                       nround = 600)

# Predict
pred <- predict(xgb.model, newdata = dtest.X)

# RMSE
sqrt(mean((pred - test.Y)^2)) 

# RMSE in kcal/mol SCALE 
sqrt(mean((pred - test.Y)^2)) *Rydberg_to_kcal_mol


# write out results
full_results = rbind(full_results, res.nsteps$par)
full_results$results = c(results_2, res.nsteps$value)


path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/energy_state_optimized.csv"
fwrite(full_results,path.out)


