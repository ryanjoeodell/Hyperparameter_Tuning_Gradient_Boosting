library(data.table)
library(xgboost)
library(lightgbm)
library(caret)
library(Rcpp)
library(lhs)

library(DiceKriging)
library(DiceOptim)
library(DiceView)

path = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/"

# load the R data files 
load(file = paste0(path,"x.train.rda"))
load(file = paste0(path,"y.train.rda"))
load(file = paste0(path,"x.test.rda"))
load(file = paste0(path,"y.test.rda"))

str(x.train)
str(x.test)

gc(reset = TRUE)

# lightgbm sudo code
train.mat = lgb.Dataset(data  = x.train ,
                        label = y.train , 
                        free_raw_data = TRUE )

# was very intense on the RAM

valid.mat = lgb.Dataset(data  = x.test,
                        label = y.test , 
                        free_raw_data = TRUE )

rm(x.train)
rm(x.test)

gc(reset = TRUE)

# configure light gbm parameters
params = list(objective = "regression",
              metric = "rmse",
              seed = 20,
              learning_rate = 0.01,
              lambda_l1 = 0.25,
              num_leaves = 32,
              max_depth = 6,
              #bagging_fraction = 0.66,
              #bagging_freq = 1, 
              colsample_bytree = 0.77,
              device = "gpu" , 
              gpu_device_id = 1L,
              gpu_use_dp= FALSE ,
              max_bin = 64L ,      # true lightgbm 
              tree_learner =  "data",
              num_thread = 0 ) # good for large data small feature 

system.time({
model_1 =  lgb.cv(params,
                  train.mat ,
                  nrounds = 100 , 
                  nfold = 5,
                  eval_freq = 100) })

model_1$record_evals$valid$rmse$eval[[100]]

# user  system elapsed 
# 1301.18  353.98  147.88 

# took like 2 minutes 

# fetch performance metric 

model_1 = lgb.train(params,
                    train.mat,
                    nrounds = 750)

# wild 
#preds = predict(model_1 , x.test )

#plot(preds,y.test)
#cor(preds,y.test)^2

#sqrt( mean( (preds - y.test)^2 ))


#### set up LHS sample
#### computer simulation
#### km modeling / optimization


### nrounds, learn_rate, leaves,bagfrac,feature fraction

set.seed(1919) 

final_design <- maximinLHS(15, 5,
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
  learn_rate   = X[1]
  bag_frac     = X[2]
  feature_frac = X[3]
  num_leaves   = X[4]
  num_rounds   = X[5]
  
  # continuous parameters
  learn_rate   = inversenormalize(learn_rate, 0.01, 0.1 )
  bag_frac     = inversenormalize(bag_frac, 0.25, 1 )
  # change it 
  feature_frac = inversenormalize(feature_frac, 0.75, 1 )
  
  
  # discrete parameters 
  num_leaves = ceiling( qunif(num_leaves, 16, 256  ) )
  num_rounds = ceiling( qunif(num_rounds, 100 , 1500 ))

  print(paste0("Learn rate: ", learn_rate  ))
  print(paste0("Bag Frac: "  , bag_frac    ))
  print(paste0("Feature Frac", feature_frac))
  print(paste0("Num Leaves:" , num_leaves  ))
  print(paste0("Learn rate:" , num_rounds  ))
  
  
  # parameter set up 
  params = list(objective = "regression",
                metric = "rmse",
                seed = 20,
                learning_rate = learn_rate,
                num_leaves = num_leaves ,
                bagging_fraction = bag_frac,
                bagging_freq = 1, 
                colsample_bytree = feature_frac,
                device = "gpu" , 
                gpu_device_id = 1L,
                gpu_use_dp= FALSE ,
                max_bin = 64L ,      # true lightgbm 
                tree_learner =  "data",
                num_thread = 0 ) # good for large data small feature 
  
  
  set.seed(100)
  
  model_2 =  lgb.cv(params,
                    train.mat ,
                    nrounds = num_rounds , 
                    nfold = 5,
                    eval_freq = num_rounds ) 
  
  performance = model_2$record_evals$valid$rmse$eval[[num_rounds]]
  
  # its RMSE so square it 
  results = performance
  
  # pause execution for a few
  # minutes to prevent overheating of GPU
  gc( reset = TRUE )
  Sys.sleep(120L)
  print(results)
  return(results)
  
}

# time it / ensure it works 
system.time({
  computer_simulation_2(as.numeric(final_design[1, 1:5]))
})

# took about 5 minutes without sys.sleep
# high on memory though 

results_2 = rep(0, 15)
system.time({
  for(i in 1:15){
    tmp = computer_simulation_2(as.numeric(final_design[i, 1:5]))
    results_2[i] = tmp 
    gc(reset = TRUE )
  }
})

full_results = data.frame(results = results_2,
                          learn_rate = final_design[,1],
                          bag_frac   = final_design[,2],
                          feature_frac   = final_design[,3],
                          num_leaves = final_design[,4],
                          num_rounds = final_design[,5])
# read in data 

path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/results_logD.csv"
fwrite(full_results,path.out)

#full_results = fread(path.out)
#results_2 <- full_results$results

# delete the results column
# and build kriging model 
full_results$results <- NULL
model = km(formula = ~ ., 
           design = full_results, 
           response = results_2,
           noise.var = rep(1e-16, 25))

# need to make the plot window huge
DiceView::sectionview(model , 
                      center = rep(0.5,5), 
                      col_points = "black")

# select best parameter as init

res.nsteps <- EGO.nsteps(model = model ,
                         fun = computer_simulation_2,
                         nsteps = 10,
                         lower = rep(0, 5),
                         upper = rep(1, 5),
                         parinit = as.numeric(full_results[25,1:5]),
                         control = list(pop.size = 500,
                                        max.generations = 100,
                                        wait.generations = 25,
                                        BFGSburnin = 100))

# write out results
full_results = rbind(full_results, res.nsteps$par)
full_results$results = c(results_2, res.nsteps$value)

#full_results = fread(path.out)
#results_2 = full_results$results
#full_results$results <- NULL

path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/logD_optimized.csv"
fwrite(full_results,path.out)


# fit the model and get out of sample predictions 
# get the scores
# 20 is still the best
full_results[20,]

load(file = paste0(path,"x.test.rda"))
load(file = paste0(path,"y.test.rda"))

learn_rate   = inversenormalize(as.numeric(full_results[20,1]), 0.01, 0.1 )
bag_frac     = inversenormalize(as.numeric(full_results[20,2]), 0.25, 1 )
# change it 
feature_frac = inversenormalize(as.numeric(full_results[20,3]), 0.75, 1 )


# discrete parameters 
num_leaves = ceiling( qunif(as.numeric(full_results[20, 4]), 16, 256  ) )
num_rounds = ceiling( qunif(as.numeric(full_results[20,5]), 100 , 1500 ))

learn_rate
bag_frac
feature_frac
num_leaves
num_rounds

# configure light gbm parameters
params = list(objective = "regression",
              metric = "rmse",
              seed = 20,
              learning_rate = learn_rate,
              #lambda_l1 = 0.25,
              num_leaves = num_leaves,
              max_depth = 20,
              bagging_fraction = bag_frac,
              #bagging_freq = 1, 
              colsample_bytree = 0.90,
              device = "gpu" , 
              gpu_device_id = 1L,
              gpu_use_dp= FALSE ,
              max_bin = 64L ,      # true lightgbm 
              tree_learner =  "data",
              num_thread = 0 ) # good for large data small feature

model_1 = lgb.train(params,
                    train.mat,
                    nrounds = num_rounds)

# predictions
preds = predict(model_1, x.test)

# plot
plot(preds , preds-y.test )

# r^2 
cor(preds, y.test )^2
# 0.7952297

# RMSE
sqrt( mean( (preds - y.test)^2 ))
# 0.5625273

