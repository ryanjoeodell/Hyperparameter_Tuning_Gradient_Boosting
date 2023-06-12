library(DiceKriging)
library(DiceOptim)
library(DiceView)

library(data.table)
library(lightgbm)
library(R.utils)
library(MaxPro)
path = "C:/Users/RyanODell/Downloads/superconduct/train.csv"
train = fread(path)

cols = names(train)[!names(train) == "critical_temp"]



inversenormalize = function(y , min_x , max_x){
  X = y * (max_x - min_x) + min_x
  return(X)
}




final_design = MaxProLHD(n = 10 , p = 4, temp0 = 1000,  nstarts = 500, itermax = 100000 )
str(final_design)
final_design$Design
final_design = final_design$Design


computer_simulation = function( X ){
  #train, 
  #test , 
  #learn_rate , 
  #max_depth,
  #min_node,
  #col_sample){
  
  learn_rate = X[1]
  max_depth  = X[2]
  min_node   = X[3]
  col_sample = X[4]
  print(paste0("Learn Rate", learn_rate))
  print(paste0("max depth", max_depth))
  print(paste0("min node", min_node))
  print(paste0("col sample", col_sample))
  
  learn_rate = inversenormalize(learn_rate, 0.01, 0.02)
  max_depth  = inversenormalize(max_depth, 15 , 25 )
  min_node   = inversenormalize(min_node, 1 , 10)
  col_sample = inversenormalize(col_sample, 0.25, 0.75 )
  
  
  min_node   = ceiling(min_node)
  max_depth  = ceiling(max_depth)
  
  # parameter set up 
  params = list(objective = "regression",
                metric = "mse",
                seed = 20,
                learning_rate = learn_rate,
                #lambda_l1 = 0.25,
                min_data_in_leaf = min_node,
                max_depth = max_depth,
                num_leaves = 100,
                bagging_seed = 100,
                bagging_fraction = 0.50,
                bagging_freq = 1, 
                colsample_bytree = col_sample,
                device = "gpu" , 
                gpu_device_id = 1L,
                gpu_use_dp= FALSE ,
                max_bin = 64L )
  
  
  # need to seed set 
  set.seed(1000)
  data.part = caret::createFolds(1:nrow(train), k = 5)
  results = c()
  for(i in 1:5){
    # lightgbm sudo code
    train.idx = data.part[[i]]
    train.mat = lgb.Dataset(data  = data.matrix(train[-train.idx, ..cols]) ,
                            label = train[["critical_temp"]][-train.idx] , 
                            free_raw_data = TRUE )
    
    # was very intense on the RAM
    valid.mat = lgb.Dataset(data  = data.matrix(train[train.idx, ..cols]) ,
                            label = train[["critical_temp"]][train.idx] , 
                            free_raw_data = TRUE )
    
    lgb_model  =  lgb.train(params,
                            data = train.mat,
                            num_boost_round = 750,
                            valids = list(valids = valid.mat),
                            verbose = 0L)
    
    performance = lgb_model$eval_valid()[[1]]$value
    
    results = c(results, performance)
  }
  
  results = mean(results)
  # pause execution for a few
  # minutes to prevent overheating of GPU
  Sys.sleep(30L)
  print(results)
  return(results)
  
}


# test 
#computer_simulation(final_design[1, c(1,4,3,2)])

results = rep(0, nrow(final_design))
system.time({
  for(i in 1:nrow(final_design)){
    tmp = computer_simulation(as.numeric(final_design[i, 1:4]))
    results[i] = tmp 
    gc(reset = TRUE )
  }
})

full_results = data.frame(results = results,
                          learn_rate = final_design[,1],
                          max_depths = final_design[,2],
                          min_node   = final_design[,3],
                          col_samples = final_design[,4])

path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/results_8.csv"
fwrite(full_results,path.out)

results
full_results = full_results[,-1]
str(full_results)
par(mfrow = c(1,1))
plot(full_results$learn_rate, results)
plot(full_results$max_depths, results)
plot(full_results$min_node, results)
plot(full_results$col_samples, results)

model = km(formula = ~ ., 
           design = full_results, 
           response = results,
           noise.var = rep(0.01, 10 ))

# need to make the plot window huge
DiceView::sectionview(model , 
                      center = c(0.5, 0.5,0.5,0.5), 
                      col_points = "black",
                      bg_blend = 0 )

res.nsteps <- EGO.nsteps(model = model ,
                         fun = computer_simulation,
                         nsteps = 10, lower = rep(0, 4), upper = rep(1, 4),
                         parinit = rep(0.5, 4),
                         control = list(pop.size = 500,
                                        max.generations = 100,
                                        wait.generations = 25,
                                        BFGSburnin = 100))

plot(res.nsteps$value)

res.nsteps$par

full_results = rbind(full_results,res.nsteps$par)
full_results$results = c(results, res.nsteps$value)
# check the simulations are indeed noisy 
full_results$results[20]
computer_simulation(full_results[20, 1:4])

path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/results_9.csv"
fwrite(full_results, path.out)


# ******************************************************* #
# ******************************************************* #
# ******************************************************* #
# ******************************************************* #
# ******************************************************* #
#
#          XGBOOST EXPERIMENTS ON SUPERCONDUCTOR DATA
#
# ******************************************************* #
# ******************************************************* #
# ******************************************************* #
# ******************************************************* #
# 



# check xgboost time for 5-fold CV
# on my 12 cores and calculate timings 
library(xgboost)


train.mat = xgb.DMatrix(data = data.matrix(train[, ..cols ]),
                        label = train[["critical_temp"]])



params <- list( objective = 'reg:squarederror',
                booster = 'gbtree',
                nthread = 12,
                min_child_weight =10,
                max_depth =25, 
                learning_rate = 0.01,
                colsample_bytree = 0.75,
                subsample =0.50,
                max_leaves = 100)

system.time({
model1 <- xgb.cv( params = params,
                  data = train.mat , 
                  nrounds = 750,
                  nfold = 5)})

#user  system elapsed 
#1571.94   90.98  227.36

# 227.36 /60
# 3.789333

# took like 4 ish minutes 

train.ind = sample(1:nrow(train), (2/3)*nrow(train))

train.mat = xgb.DMatrix(data = data.matrix(train[train.ind, ..cols ]),
                        label = train[["critical_temp"]][train.ind])

test.mat = xgb.DMatrix(data = data.matrix(train[-train.ind, ..cols ]),
                        label = train[["critical_temp"]][-train.ind])

gc( reset = TRUE )

system.time({
  model2 = xgb.train(params = params,
                     data = train.mat , 
                     nrounds = 750,
                     watchlist = list(test = test.mat ))
})
#user  system elapsed 
#252.29   16.63   39.25 

# computer simulation function 
computer_simulation_2 = function( X ){

  
  learn_rate = X[1]
  max_depth  = X[2]
  min_node   = X[3]
  col_sample = X[4]
  print(paste0("Learn Rate", learn_rate))
  print(paste0("max depth", max_depth))
  print(paste0("min node", min_node))
  print(paste0("col sample", col_sample))
  
  learn_rate = inversenormalize(learn_rate, 0.01, 0.02)
  max_depth  = inversenormalize(max_depth, 15 , 25 )
  min_node   = inversenormalize(min_node, 1 , 10)
  col_sample = inversenormalize(col_sample, 0.25, 0.75 )
  
  
  min_node   = ceiling(min_node)
  max_depth  = ceiling(max_depth)
  
  # parameter set up 
  params <- list( objective = 'reg:squarederror',
                  booster = 'gbtree',
                  nthread = 12,
                  min_child_weight = min_node,
                  max_depth = max_depth, 
                  learning_rate = learn_rate,
                  colsample_bytree = col_sample ,
                  subsample =0.50,
                  max_leaves = 100)
  

  results = c()
  for(i in 1:25){
    set.seed(i)
    # random sample 
    # dont set seed, I want it to be randomized
    # every iteration of the loop  
    train.ind = sample(1:nrow(train), (2/3)*nrow(train))
    
    train.mat = xgb.DMatrix(data = data.matrix(train[train.ind, ..cols ]),
                            label = train[["critical_temp"]][train.ind])
    
    test.mat = xgb.DMatrix(data = data.matrix(train[-train.ind, ..cols ]),
                           label = train[["critical_temp"]][-train.ind])
    
    model2 = xgb.train(params = params,
                       data = train.mat , 
                       nrounds = 750,
                       watchlist = list(test = test.mat ))
    
    performance = model2$evaluation_log[750, test_rmse]
    
    # its RMSE so square it 
    results = c(results, performance**2 )
    gc( reset = TRUE )
  }
  
  results = mean(results)
  # pause execution for a few
  # minutes to prevent overheating of GPU
  Sys.sleep(120L)
  print(results)
  return(results)
  
}

# time it / ensure it works 
system.time({
computer_simulation_2(as.numeric(final_design[i, 1:4]))
})
# [1] 87.87635 with seeds set
# 87.87635 exactly again

# [1] 87.13548
#  user  system elapsed 
# 3008.58  259.86  586.38 

results_2 = rep(0, 10)
system.time({
  for(i in 1:10){
    tmp = computer_simulation_2(as.numeric(final_design[i, 1:4]))
    results_2[i] = tmp 
    gc(reset = TRUE )
  }
})

# ensure enlightment with values
# passed to 
full_results_2 =
             data.frame(results = results_2,
             learn_rate = final_design[,1],
             max_depths = final_design[,2],
             min_node   = final_design[,3],
             col_samples = final_design[,4])

path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/xgboost_results1.csv"
fwrite(full_results_2 , path.out)

# set up ego code for tomorrow
library(DiceKriging)
library(DiceOptim)
library(DiceView)
full_results_2$results <- NULL
model = km(formula = ~ ., 
           design = full_results_2, 
           response = results_2,
           noise.var = rep(0.01, 10 ))

# need to make the plot window huge
DiceView::sectionview(model , 
                      center = c(0.5, 0.5,0.5,0.5), 
                      col_points = "black",
                      bg_blend = 0 )

# select best parameter as init

res.nsteps <- EGO.nsteps(model = model ,
                         fun = computer_simulation_2,
                         nsteps = 10,
                         lower = rep(0, 4),
                         upper = rep(1, 4),
                         parinit = rep(0.5, 4),
                         control = list(pop.size = 500,
                                        max.generations = 100,
                                        wait.generations = 25,
                                        BFGSburnin = 100))

# write out results
full_results_2 = rbind(full_results_2, res.nsteps$par)
full_results_2$results = c(results_2, res.nsteps$value)


path.out = "C:/Users/RyanODell/Documents/UCLA_higgs_experiment/xgboost_EGO.csv"
fwrite(full_results_2, path.out)


# modeling stuff
results_2 = full_results_2$results
full_results_2$results <- NULL
model_2 = km(formula = ~., 
           design = full_results_2, 
           response = results_2,
           covtype = "powexp",
           multistart = 10)
coef(model_2)

DiceView::sectionview(model_2 , 
                      center = rep(0.5,4) , 
                      col_points = "black",
                      bg_blend = 0 )
X = seq(0, 1 , length.out = 100)
preds = predict(model_2 , newdata = data.frame(learn_rate = X) , type = "UK")
par(mfrow = c(1,1))

plot(X, preds$mean , type ="p", pch = 16)
lines(X, preds$upper95)
lines(X, preds$lower95)

# make a dense grid 
grid = expand.grid(learn_rate = seq(0,1, length.out = 10),
                   max_depths = seq(0,1, length.out = 10),
                   min_node = seq(0,1, length.out = 10),
                   col_samples = seq(0,1, length.out = 10))
preds = simulate(model_2 , newdata = grid , type = "UK", cond = TRUE)

plot(grid$learn_rate, preds$mean)
