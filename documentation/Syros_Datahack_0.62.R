
### Title : A solution for the Rafael Challange in Datahack 2017
### author: Shay Yaacoby


## clear workspace
rm(list = ls())

## install packages if you havent already
## (h2o framework requires java runtime, see: https://www.h2o.ai/)
# install.packages("moments","tidyverse","h2o")

## read data from the compressed csv files
library(tidyverse)
tr <- read_csv(file = "train.csv.gz")

### should we use - for demo purposes - only a small part for the test data?
### replace with FALSE for evaluation on the complete test set 
use_part_of_test_data = TRUE
tst <- read_csv(file = "test.csv.gz",n_max = if (use_part_of_test_data) 5000 else Inf)

glimpse(tr)
glimpse(tst)

## combine train and test; reshape the data to longer format (row for each time point);
## sort by time points
x_long <- bind_rows(tr = tr, tst = tst, .id = "dataset") %>%
  gather(key = "key",value = "value", - X1 ,- targetName, - class, -dataset) %>%
  separate(col = key, into = c("measure", "point"), sep = "_") %>%
  spread(key = measure, value = value) %>%
  filter(!is.nan(Time)) %>%
  arrange(dataset, X1, Time)

rm(tst,tr)

## some initial univariate exploratory visualizations (presented for each for class)
theme_set(theme_minimal())
ggplot( x_long %>% filter(dataset == "tr"), aes(posZ)) + 
  geom_histogram() + 
  facet_wrap(~ class,scales = "free_y") +
  scale_x_continuous(breaks=seq(0, 100000, 20000), labels = seq(0, 100000, 20000)/1000) +
  scale_y_continuous(breaks=NULL, labels = NULL) +
  ggtitle("distribution of vertical positions (scaled for each class)") +
  xlab("vertical position (posZ/1000)")

## compute variables: horizonal and total lengths of the position and velocity vectors
x_long <- x_long %>%
  mutate(
    posH = sqrt(posX^2 + posY^2),
    posA = sqrt(posH^2 + posZ^2),       
    velH = sqrt(velX^2 + velY^2),
    velA = sqrt(velH^2 + velZ^2)) 

## compute changes (delta) in the position and velocity vectors (in both coordinate systems), 
## the angel changes
x_long <- x_long %>%   
  group_by(dataset,X1,targetName,class) %>% 
  mutate(posdiffX = posX - lag(posX),
         posdiffY = posY - lag(posY),
         posdiffZ = posZ - lag(posZ),
         
         veldiffX = velX - lag(velX),
         veldiffY = velY - lag(velY),
         veldiffZ = velZ - lag(velZ),
         
         posdiffH = sqrt(posdiffX^2 + posdiffY^2),
         veldiffH = sqrt(veldiffX^2 + veldiffY^2),
         
         posdiffA = sqrt(posdiffX^2 + posdiffY^2 + posdiffZ^2),
         veldiffA = sqrt(veldiffX^2 + veldiffY^2 + veldiffZ^2),
         
         relangelH = atan(posdiffY/posdiffX),
         relangelA = atan(posdiffZ/posdiffH))
         
# ggplot(data = x_long %>% filter(dataset == "tr")) + 
#   aes(x = posdiffH, y = posdiffZ) + 
#   geom_hex() + 
#   facet_wrap(~ class)#, scales = "free")

## for each trajectory compute the mean, min, max, sd, skewness of the computed variables.
library(moments)
x_wide1 <- x_long %>% group_by(dataset, class, targetName, X1) %>% 
  summarise_at(vars(posZ,posH,velZ,velH,velA, posdiffH, veldiffH, posdiffA, veldiffA, relangelH, relangelA),
  funs(min , max, mean, sd ,skewness), na.rm = T) %>% 
## also add 0,1 variable for the tragectory direction (up\dowm)
  mutate(sign_relangelA_mean = sign(relangelA_mean))

## also compute the mean of the absolute value of the angel changes
x_wide2 <- x_long %>% group_by(dataset, class, targetName, X1) %>% 
  summarise_at(vars(veldiffA, relangelH), funs(amean = mean(abs(.))), na.rm = T)

## get the coefficients of a polynomial fit on each trajectory
fit_poly <- function(x,y) lm(y ~ x + I(x^2))$coefficients %>% t() %>% data.frame()
x_wide3 <- x_long %>%
  group_by(dataset, class, targetName, X1) %>%
  do(bind_cols(fit_poly(.$posH,.$posZ),fit_poly(.$velH,.$velZ)))

## merge the 3 datasets
x <- x_wide1 %>% 
  inner_join(x_wide2) %>% 
  inner_join(x_wide3) %>% 
  group_by(class) %>% 
  select(-X1, -posH_min)

## verify that there are no NAs and write to csv file
x %>% nrow()
x %>% filter_all(any_vars(. != 0)) %>% nrow()

write_csv(x,"x.csv")

# -----------------------------------------------------------------------

## clear workspace for more memory, start h2o and load dataset to the h2o "cluster"
rm(list = ls())
library(h2o)
h2o.shutdown(F)
h2o.init()
xh <- h2o.importFile(path = "x.csv", destination_frame = "x")
xh$class <- as.factor(xh$class)

## train pca (on the complete dataset, train+test) 
pca_model <- 
  h2o.prcomp(training_frame = xh, model_id = "pca1", k = 10, transform = "STANDARDIZE",
             seed = 1, x = 4:ncol(xh))

## plot the values
plot(pca_model@model$importance[1,] %>% t())

## => take the first 8 PC
x_pcs <- predict(pca_model,xh)
xh <- h2o.cbind(xh,x_pcs)

## train the model as in Datahack 2017 
## this is xgboost model
gbm <- h2o.gbm(y = "class", x = 4:ncol(xh), # x = varimp_tbl$variable[1:30],
               training_frame = xh[xh$dataset == "tr",],
               ntrees = 150, # in DH were 300
               max_depth = 10, 
               seed = 1,
               nfolds = 5,
               min_rows = 5,
               learn_rate = .1)

## model *accuracy* (h2o framework deprecated the f1-score metric):
gbm@model$cross_validation_metrics_summary$mean[1] # ==> 0.63416
## with ntrees=300 ==> 0.63864833 

## variable importance:
varimp_tbl <- h2o.varimp(gbm)
save(varimp_tbl,file = "varimp_tbl.RData")

## plot the relative importance
h2o.varimp_plot(gbm,num_of_features = 20)

## generate prediction:
predh <- h2o.predict(gbm, xh[xh$dataset!="tr", 4:ncol(xh)])
## write_csv(predh %>% as.data.frame(),"pred_demo.csv",col_names = F)

## import from h2o to R; reshape for submission and write to csv file
pred <- predh[1] %>% as.data.frame() %>% as.tibble()
submit <- pred %>% rowid_to_column() %>% mutate(rowid = rowid - 1) 
write_csv(submit,"submission_demo.csv",col_names = F)

# sanity check for the structure of the submission
submit %>% pull(predict) %>% table()
glimpse(submit)

## **END**