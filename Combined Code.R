## Install new packages
install.packages( "fastAdaboost")
#------------------------------------------
## Load libraries
library(rpart) # decision trees
library(rpart.plot) # decision tree plots
library(caret) # machine learning
library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra)
library(dplyr)
library(e1071)
library(randomForest)

library(caretEnsemble) # training multiple models, custom ensembles
library(parallel) # parallel processing
library(doParallel) # parallel processing
library(fastAdaboost)
library(caret)
library(PreProcess)
library(cluster)
library(htestClust)
library(Rtsne)
library(fpc)
###############################################################
#########         CLUSTERING ANALYSIS                ##########
###############################################################


data_comb <- read.csv('Combined_sampled_data_dscp.csv')
#Duplicating the dataset
data_combnew1 <- data_comb
#Changing the Age into Age Groups
data_combnew1$AGEGrp[data_combnew1$AGE<=30] <-1
data_combnew1$AGEGrp[data_combnew1$AGE>=31 & data_combnew1$AGE<=40] <-2
data_combnew1$AGEGrp[data_combnew1$AGE>=41 & data_combnew1$AGE<=50] <-3
data_combnew1$AGEGrp[data_combnew1$AGE>=51 & data_combnew1$AGE<=60] <-4
data_combnew1$AGEGrp[data_combnew1$AGE>=60] <-5
table(data_combnew1$AGEGrp)

#Creating vector for categorical variables
cats_comb2 <- c('EGENDER','EEDUC','MS','INCOME','EST_ST','REGION','RHISPANIC','RRACE',
                'EXPCTLOSS','ANYWORK','TW_YN','UI_APPLYRV','WRKLOSSRV','ANXIOUS','WORRY',
                'INTEREST','DOWN','PRESCRIPT','MH_SVCS','MH_NOTGET','SSA_RECV','SSA_APPLYRV',
                'HADCOVID','AGEGrp','RECVDVACC')

#Creating vector for numerical variable
nums_comb2 <- c('THHLD_NUMKID','THHLD_NUMADLT','TNUM_PS')

#Removing and adding RECVDVACC
data_combnew2 <- data_combnew1[,-30]
data_combnew2$RECVDVACC <- data_combnew1$RECVDVACC
data_combnew2$RECVDVACC <- as.factor(data_combnew2$RECVDVACC)
data.frame(colnames(data_combnew2))

#Removing redundant variables
#X, WEEK, SCRAM, PHASE
data_combnew3 <- data_combnew2[,-c(1,9,20,31)]

#Factorize the categorical variables
data_combnew3[,cats_comb2] <- lapply(X = data_combnew3[,cats_comb2], FUN = factor)

#converting integer variables into Numeric
data_combnew3[,nums_comb2] <- lapply(X = data_combnew3[,nums_comb2], FUN = as.numeric)

#Removing Age variable as it has already been converted into groups
data_combnew4 <- data_combnew3[,-8]

#Scaling
cen_cc <-preProcess(x=data_combnew4, method = c('center','scale'))
data_combnew4 <- predict(object = cen_cc,
                         newdata = data_combnew4)

set.seed(2969)

#Calculating gower distance
hdist2_combnew4 <- daisy(x = data_combnew4, metric= "gower")
#Clustering
wards1_combnew4 <- hclust(d = hdist2_combnew4, 
                          method = "ward.D2")

#Load Clustering.RData file for sil and wss plot functions
load("~/Desktop/Capstone/Data/Clustering.RData")
wss_plot(dist_mat = hdist2, # distance matrix
         method = "hc", # HCA
         hc.type = "ward.D2", # linkage method
         max.k = 30) # maximum k value
## Strict Elbow at k =  3

sil_plot(dist_mat = hdist2_combnew4, # distance matrix
         method = "hc", # HCA
         hc.type = "ward.D2", # average linkage
         max.k = 30) # maximum k value

#Reducing the dimensionality using Rtsne function from Rtsne package
ld_dist_combnew4 <- Rtsne(X = hdist2_combnew4, 
                          is_distance = TRUE)

lddf_dist_combnew4 <- data.frame(ld_dist_combnew4$Y)

#Creating the tree
wards1_clusters_combnew4 <- cutree(tree = wards1_combnew4, k = 3)

# Plotting the clusters
ggplot(data = lddf_dist_combnew4, 
       mapping = aes(x = X1, y = X2)) +
  geom_point(aes(color = factor(wards1_clusters_combnew4))) +
  labs(color = "Cluster")

table(wards1_clusters_combnew4)

#Describing cluster solutions for wards method
aggregate(x = data_combnew4[ ,nums_comb2], 
          by = list(wards1_clusters_combnew4),
          FUN = mean)

cluster_result_combnew4 <- data.frame(aggregate(x = data_combnew4[ ,cats_comb2], 
                                                by = list(wards1_clusters_combnew4),
                                                FUN = table))

#Saving the cluster results
write.csv(cluster_result_combnew4,'cluster_result_combnew4_latest.csv')

## Cluster Validation
#Adjusted Rand Index (should be between -1 to 1) #not so good not so bad
cluster.stats(d = hdist2_combnew4, # distance matrix
              clustering = wards1_clusters_combnew4, # cluster assignments
              alt.clustering = as.numeric(data_combnew4$RECVDVACC))$corrected.rand 

#Cophonetic Distance #Goodness of fit
# Ward's Method
cor(x = hdist2_combnew4, y = cophenetic(x = wards1_combnew4))

# Obtain Cluster Statistics
stats_HCA_combnew4 <- cluster.stats(d = hdist2_combnew4, 
                                    clustering = wards1_clusters_combnew4)
stats_HCA_combnew4

######################
# combine data from phase 3

w22 <- read.csv('pulse2021_puf22.csv') #weel 22 survey data
w23 <- read.csv('pulse2021_puf23.csv') #week 23 survey data
w24 <- read.csv('pulse2021_puf24.csv') #week 24 survey data
w25 <- read.csv('pulse2021_puf25.csv') #week 25 survey data
w26 <- read.csv('pulse2021_puf26.csv') #week 26 survey data
w27 <- read.csv('pulse2021_puf27.csv') #week 27 survey data

p3 <- rbind.fill(w22,w23,w24,w25,w26,w27) #combine all survey data from phase 3

write.csv(p3, 'Week22-27Data.csv')

# combine data from phase 3.1

w28 <- read.csv('pulse2021_puf28.csv') #weel 28 survey data
w29 <- read.csv('pulse2021_puf29.csv') #week 29 survey data
w30 <- read.csv('pulse2021_puf30.csv') #week 30 survey data
w31 <- read.csv('pulse2021_puf31.csv') #week 31 survey data
w32 <- read.csv('pulse2021_puf32.csv') #week 32 survey data
w33 <- read.csv('pulse2021_puf33.csv') #week 33 survey data

p3.1 <- rbind.fill(w28,w29,w30,w31,w32,w33) #combine all survey data from phase 3

write.csv(p31., 'Week27-33Data.csv')

#*################################****#
# open week 22-27 data
p3 <- read.csv('Week22-27Data.csv')
# open week 28-33 data
p3.1 <- read.csv('Week27-33Data.csv')
###----------------------------------------------------
## recode missing data (-99 and -88 into na)
p3_v1 <- p3
p3_v1 <- p3_v1 %>% mutate_at(vars(colnames(p3_v1)), na_if, -88)
p3_v1 <- p3_v1 %>% mutate_at(vars(colnames(p3_v1)), na_if, -99)
head(p3_v1)
p3.1_v1 <- p3.1
p3.1_v1 <- p3.1_v1 %>% mutate_at(vars(colnames(p3.1_v1)), na_if, -88)
p3.1_v1 <- p3.1_v1 %>% mutate_at(vars(colnames(p3.1_v1)), na_if, -99)
head(p3.1_v1)
## retain only those answered "GETVACC" (P3) with 1,2,3,4
p3_v1 <- p3_v1[p3_v1$GETVACC %in% c(1,2,3,4),]
## retain only those answered "GETVACRV" (P3.1) with 1,2,4,5 (excluding3)
p3.1_v1 <- p3.1_v1[p3.1_v1$GETVACRV %in% c(1,2,4,5),]
###----------------------------------------------------
# Get data category, primary, consistent info
p3_summary <- read.csv('HPS_p3_datatype_category_PRED.csv')
p3_summary
p3.1_summary <- read.csv('HPS_p3.1_datatype_category_PRED.csv')
p3.1_summary
###----------------------------------------------------
## vectors for consistent variables (include secondary variable for DT, NB)
p3_var_keep <- p3_summary$Variable[p3_summary$Consistent %in% 
                                     c(1,"Var-name-diff", 
                                       "Need-recode")]
#& p3_summary$Primary.Q == 1]
p3.1_var_keep <- p3.1_summary$Variable[p3.1_summary$Consistent %in% 
                                         c(1,"Var-name-diff", 
                                           "Need-recode")]
#& p3.1_summary$Primary.Q == 1]
setdiff(p3_var_keep, p3.1_var_keep)
## vectors for primary var
p3_prim <- p3_summary$Variable[p3_summary$Primary.Q == 1 & 
                                 p3_summary$Consistent %in% 
                                 c(1,"Var-name-diff", 
                                   "Need-recode")]
p3.1_prim <- p3.1_summary$Variable[p3.1_summary$Primary.Q == 1 & 
                                     p3.1_summary$Consistent %in% 
                                     c(1,"Var-name-diff", 
                                       "Need-recode")]
###----------------------------------------------------
## Study missingness of datasets with consistent vars only
## Phase 3
p3_chk <- p3_v1[c(p3_var_keep)]
p3_chk$missing_pct <- rowSums(is.na(p3_chk))/ncol(p3_chk)*100
summary(p3_chk$missing_pct)
nrow(p3_chk[p3_chk$missing_pct < 50,])/nrow(p3_chk)
p3_chk$missing_prim_pct <- rowSums(is.na(p3_chk[p3_prim]))/ncol(p3_chk)*100
summary(p3_chk$missing_prim_pct)
nrow(p3_chk[p3_chk$missing_prim_pct < 20,])/nrow(p3_chk)
# missingness of variables
colSums(is.na(p3_chk))/nrow(p3_chk)*100
colSums(is.na(p3_chk[p3_prim]))/nrow(p3_chk)*100
## Phase 3.1
p3.1_chk <- p3.1_v1[p3.1_var_keep]
p3.1_chk$missing_pct <- rowSums(is.na(p3.1_chk))/ncol(p3.1_chk)*100
summary(p3.1_chk$missing_pct)
nrow(p3.1_chk[p3.1_chk$missing_pct < 50,])/nrow(p3.1_chk)
p3.1_chk$missing_prim_pct <- rowSums(is.na(p3.1_chk[p3.1_prim]))/ncol(p3.1_chk)*100
summary(p3.1_chk$missing_prim_pct)
nrow(p3.1_chk[p3.1_chk$missing_prim_pct < 20,])/nrow(p3.1_chk)
# missingness of variables
colSums(is.na(p3.1_chk))/nrow(p3.1_chk)*100
colSums(is.na(p3.1_chk[p3.1_prim]))/nrow(p3.1_chk)*100
###----------------------------------------------------
## Save datasets of consistent variables, including secondary var, separately
## Phase 3
p3_keep <- p3_v1[c(p3_var_keep)]
p3_keep <- p3_keep[!colnames(p3_keep) %in% c('EST_MSA', 'HLTHINS1','HLTHINS2'
                                             ,'HLTHINS3','HLTHINS4','HLTHINS5'
                                             ,'HLTHINS6','HLTHINS7', 'HLTHINS8'
                                             ,'SPNDSRC1','SPNDSRC2',	'SPNDSRC3'
                                             ,'SPNDSRC4','SPNDSRC5','SPNDSRC6'
                                             ,'SPNDSRC7', 'SPNDSRC8')]
## Phase 3.1
p3.1_keep <- p3.1_v1[p3.1_var_keep]
p3.1_keep <- p3.1_keep[!colnames(p3.1_keep) %in% c('EST_MSA', 'HLTHINS1'
                                                   ,'HLTHINS2'
                                                   ,'HLTHINS3','HLTHINS4','HLTHINS5'
                                                   ,'HLTHINS6','HLTHINS7', 'HLTHINS8'
                                                   ,'SPNDSRC1','SPNDSRC2',	'SPNDSRC3'
                                                   ,'SPNDSRC4','SPNDSRC5','SPNDSRC6'
                                                   ,'SPNDSRC7', 'SPNDSRC8')]
###----------------------------------------------------
## Change var name in p3 to match p3.1
p3_keep <- p3_keep %>% rename(WRKLOSSRV = WRKLOSS
                              , UI_APPLYRV = UI_APPLY
                              , SSA_APPLYRV = SSA_APPLY
                              , SSAPGMRV1 = SSAPGM1
                              , SSAPGMRV2 = SSAPGM2
                              , SSAPGMRV3 = SSAPGM3
                              , SSAPGMRV4 = SSAPGM4
                              , SSAPGMRV5 = SSAPGM5
                              , SSALIKELYRV = SSALIKELY
                              , FOODRSNRV1 = FOODSUFRSN1
                              , FOODRSNRV2 = FOODSUFRSN2
                              , FOODRSNRV3 = FOODSUFRSN3
                              , FOODRSNRV4 = FOODSUFRSN4
                              , INTRNTRV1 = INTRNT1
                              , INTRNTRV2 = INTRNT2
                              , INTRNTRV3 = INTRNT3)
###----------------------------------------------------
## recode some p3 variables to match with p3.1
# p3 TW_START: 1=1, 2/3 =2 to match with p3.1
p3_keep$TW_START[p3_keep$TW_START == 3] <- 2
p3_keep <- p3_keep %>% rename(TW_YN = TW_START)
# p3 EIP: 1/2/3 = 1; 4 = 2 to match with EIP_YN
p3_keep$EIP[p3_keep$EIP %in% c(1,2,3)] <- 1
p3_keep$EIP[p3_keep$EIP == 4 ] <- 2
table(p3_keep$EIP)
p3_keep <- p3_keep %>% rename(EIP_YN = EIP)
# p3 LIVQTR: 6/7/8/9 = 6; 10 = 7
table(p3_keep$LIVQTR)
p3_keep$LIVQTR[p3_keep$LIVQTR %in% c(6,7,8,9)] <- 6
p3_keep$LIVQTR[p3_keep$LIVQTR == 10 ] <- 7
p3_keep <- p3_keep %>% rename(LIVQTRRV = LIVQTR)
table(p3_keep$LIVQTRRV)
# p3 RSNNOWRK: 1=1; 2/3=2; 4=3; 5=4; 6=6; 7=7, 8/9=8; 10=9; 11=10; 13=5; 12=12
table(p3_keep$RSNNOWRK)
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK %in% c(2,3)] <- 2
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK == 4 ] <- 3
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK == 5 ] <- 4
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK %in% c(8,9)] <- 8
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK == 10 ] <- 9
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK == 11 ] <- 10
p3_keep$RSNNOWRK[p3_keep$RSNNOWRK == 13 ] <- 5
p3_keep <- p3_keep %>% rename(RSNNOWRKRV = RSNNOWRK)
table(p3_keep$RSNNOWRKRV)
## recode p3.1 to match p3
# p3.1 GETVACRV: 1=1; 2=2; 4=3; 5=4
# GETVACRV = GETVACC
table(p3.1_keep$GETVACRV)
table(p3_keep$GETVACC)
p3.1_keep$GETVACRV[p3.1_keep$GETVACRV == 4] <- 3
p3.1_keep$GETVACRV[p3.1_keep$GETVACRV == 5] <- 4
table(p3.1_keep$GETVACRV)
p3_keep <- p3_keep %>% rename(GETVACRV = GETVACC)
# p3.1 TENROLLPUB > 0 OR TENROLLPRV > 0 --> ENROLL1 = 1
table(p3.1_keep$TENROLLPUB)
table(p3.1_keep$TENROLLPRV)
p3.1_keep$ENROLL1 <- NA
p3.1_keep[p3.1_keep$TENROLLPUB %in% c(1,2,3,4), 'ENROLL1'] <- 1
p3.1_keep[p3.1_keep$TENROLLPRV %in% c(1,2), 'ENROLL1'] <- 1
table(p3.1_keep$ENROLL1)
# p3.1 TENROLLHMSCH > 0 --> ENROLL2 = 1
table(p3.1_keep$TENROLLHMSCH)
p3.1_keep$TENROLLHMSCH[p3.1_keep$TENROLLHMSCH %in% c(1,2) ] <- 1
p3.1_keep <- p3.1_keep %>% rename(ENROLL2 = TENROLLHMSCH)
# p3.1 ENROLLNONE = 1 --> ENROLL3 = 1
table(p3.1_keep$ENROLLNONE)
p3.1_keep <- p3.1_keep %>% rename(ENROLL3 = ENROLLNONE)
table(p3.1_keep$ENROLL3)
# find all values in char1 that do not occur in char2
setdiff(colnames(p3_keep), colnames(p3.1_keep))
setdiff(colnames(p3.1_keep),colnames(p3_keep)) # "TENROLLPUB", "TENROLLPRV"
# remove "TENROLLPUB", "TENROLLPRV" in p3.1_keep
p3.1_keep <- p3.1_keep[!colnames(p3.1_keep) %in% c("TENROLLPUB", "TENROLLPRV")]
###----------------------------------------------------
## add a column to indicate phase
p3_keep$PHASE <- 'P3'
p3.1_keep$PHASE <- 'P3.1'
## factorize nominal and ordinal var
nom <- p3.1_summary$Variable[p3.1_summary$Var.type == 'nom' &
                               p3.1_summary$Variable %in% colnames(p3.1_keep) &
                               !p3.1_summary$Variable %in% c('TENROLLPUB', 'TENROLLPRV'
                                                             ,'TENROLLHMSCH','ENROLLNONE')]
nom <- c(nom, 'ENROLL1','ENROLL2','ENROLL3')
num <- p3.1_summary$Variable[p3.1_summary$Var.type == 'num' &
                               p3.1_summary$Variable %in% colnames(p3.1_keep)]
ord <- p3.1_summary$Variable[p3.1_summary$Var.type == 'ord' &
                               p3.1_summary$Variable %in% colnames(p3.1_keep)]

###----------------------------------------------------
## Combining p3 and p3.1
cmb_keep <- bind_rows(p3_keep, p3.1_keep)
setdiff(colnames(cmb_keep), colnames(p3.1_keep))
setdiff(colnames(cmb_keep), colnames(p3_keep))
head(cmb_keep)
tail(cmb_keep)
write.csv(cmb_keep, 'combined_full_data_PRED.csv') # THIS WAS USED IN PREDICTION ANALYSIS 
############for Descriptive Model 

p3 <- read.csv("Week22-27Data.csv", header=TRUE)
p3 <- p3 %>% mutate_at(vars(colnames(p3)), na_if, -88)
p3 <- p3 %>% mutate_at(vars(colnames(p3)), na_if, -99)


p3$GETVACC[p3$GETVACC == 2] <- 1 #change GETVAC from 2 to 3
p3$GETVACC[p3$GETVACC == 4] <- 0 #change GETVAC from 4 to 5
p3$GETVACC[p3$GETVACC == 3] <-0



p31<- read.csv("Week27-33Data.csv", header=TRUE)
p31 <- rename(p31, c(
  'GETVACC'='GETVACRV'))
  
 # 'FOODRSNRV1'='FOODSUFRSN1',
 # 'FOODRSNRV2'='FOODSUFRSN2',
  #'FOODRSNRV3'='FOODSUFRSN3',
  #'FOODRSNRV4'='FOODSUFRSN4',
  #'SSAPGMRV1' ='SSAPGM1',
  #'SSAPGMRV2' ='SSAPGM2',
  #'SSAPGMRV3' ='SSAPGM3',
  #'SSAPGMRV4' ='SSAPGM4',
  #'SSAPGMRV5' ='SSAPGM5',
  #'SSALIKELYRV'='SSALIKELY',
  #'RSNNOWRKRV'='RSNNOWRK')
#) #change some variables name so it can be consistent

p31 <- p31 %>% mutate_at(vars(colnames(p31)), na_if, -88)
p31 <- p31 %>% mutate_at(vars(colnames(p31)), na_if, -99)
#p31 <- na.omit(p31)
#change GETVACC 
p31$GETVACC[p31$GETVACC == 2] <- 1
p31$GETVACC[p31$GETVACC == 4] <- 0
p31$GETVACC[p31$GETVACC == 3] <- 0
p31$GETVACC[p31$GETVACC == 5] <- 0

dp   <- rbind.fill(p3,p31) #merged phase 3 and 3.1

dp1 <- subset (dp, select = c("EGENDER", "EEDUC",
                                "MS", "THHLD_NUMKID",
                                "INCOME" ,
                                "AGE" ,"RHISPANIC",
                                "RRACE","THHLD_NUMADLT",
                                "TNUM_PS", "EXPCTLOSS",
                                "ANYWORK", "TW_YN",
                                "UI_APPLYRV", "WRKLOSSRV",
                                "CURFOODSUF" ,"FREEFOOD",
                                "TSPNDFOOD","TSPNDPRPD",
                                "SNAP_YN", "DELAY",
                                "NOTGET" ,"PRIVHLTH",
                                "PUBHLTH", "TENURE",
                                "LIVQTRRV", "ANXIOUS",
                                "WORRY" ,"INTEREST",
                                "DOWN" ,"PRESCRIPT",
                                "MH_SVCS" ,"MH_NOTGET",
                                "SSA_RECV" ,"SSA_APPLYRV",
                                "EIP_YN" ,"EXPNS_DIF","HADCOVID"
                                ))
# set Default variable
dp1 <- na.omit(dp1)
dp1$ANXIOUS <- factor (dp1$ANXIOUS)
summary(dp1$ANXIOUS)

#mental health prescription based on gender
dp1$AgeBin <- cut (dp1$AGE, breaks = c (0,30,40,50,60,70,80,100),
                             labels = c('18-30','30-40','40-50','50-60','60-70','70-80','80 and older'))


#people with more than 1 kid
kid <- subset(dp1, THHLD_NUMKID >0)


#age with down
dp1$ANXIOUS[dp1$ANXIOUS == 4] <- 2
dp1$ANXIOUS[dp1$ANXIOUS == 3] <- 2

summary(dp1$ANXIOUS)
 plot(dp1$ANXIOUS,
       main = "Frequency of Feeling Anxiety", col = c("#0072B2","#E69F00"),
        ylab = "Count", pch = 19)
 legend( "topright", 
         legend = c("Did not have Anxiety", "Feel Anxiety"),lwd = 2, col = c('#0072B2', '#E69F00'))

dp1$DOWN <- factor (dp1$DOWN)
ggplot(data = dp1, aes(x = AgeBin, fill = ANXIOUS)) +
#  geom_bar(position = "dodge")+
  geom_bar(stat = "identity")+
  labs(x = "AGE RANGE", y = "Count", title = "Frequency of Feeling Anxiety") +
  scale_fill_discrete(labels = c("YES", "NO"))+
#  scale_x_discrete(labels=c("1" = "Yes", "2" = "No"
 #                           ))+
  
  theme_minimal()


table(dp1$AgeBin, dp1$ANXIOUS)

#how to ##anxiety 
anxiety <- subset(dp1, ANXIOUS = 2)
anxiety$EXPNS_DIF <- factor(anxiety$EXPNS_DIF)
anxiety$THHLD_NUMKID[anxiety$THHLD_NUMKID == 2] <- 1
anxiety$THHLD_NUMKID[anxiety$THHLD_NUMKID == 3] <- 1
anxiety$THHLD_NUMKID[anxiety$THHLD_NUMKID == 4] <- 1

anxiety$EXPNS_DIF[anxiety$EXPNS_DIF == 3] <- 2
anxiety$EXPNS_DIF[anxiety$EXPNS_DIF == 4] <- 2
anxiety$AgeBin <- cut (anxiety$AGE, breaks = c (0,30,40,50,60,70,80,100),
                   labels = c('18-30','30-40','40-50','50-60','60-70','70-80','80 and older'))



nums <- c("NUMADLT",'TNUM_PS','THHLD_NUMKID','THHLD_NUMADLT','AGE','TSPNDPRPD',
          'TSPNDFOOD')
noms <- names(anxiety)[!names(anxiety) %in% c(nums)]
## Convert Nominal Variables
df[ ,noms] <- lapply(X = anxiety[ ,noms], 
                     FUN = factor)

ggplot(data = anxiety, aes(x = AgeBin, fill = ANXIOUS)) +
  geom_bar(position = "dodge")+ geom_bar( stat="identity") 

summary(anxiety$AgeBin)
  
##################### FOR PREDICTION MODEL
model <- read.csv("Combined_full_data_PRED.csv", header=TRUE)
#View(model)
# set Default variable
#model$GETVACRV <- factor(model$GETVACRV)
summary(model$GETVACRV)

df <- subset (model, select = c("EGENDER", "EEDUC",
                                 "MS", "THHLD_NUMKID",
                                "INCOME" ,
                                "AGE" ,"RHISPANIC",
                                "RRACE","THHLD_NUMADLT",
                                 "TNUM_PS", "EXPCTLOSS",
                                 "ANYWORK", "TW_YN",
                                "UI_APPLYRV", "WRKLOSSRV",
                                 "CURFOODSUF" ,"FREEFOOD",
                                "TSPNDFOOD","TSPNDPRPD",
                                 "SNAP_YN", "DELAY",
                                 "NOTGET" ,"PRIVHLTH",
                                 "PUBHLTH", "TENURE",
                                 "LIVQTRRV", "ANXIOUS",
                                 "WORRY" ,"INTEREST",
                                 "DOWN" ,"PRESCRIPT",
                                 "MH_SVCS" ,"MH_NOTGET",
                                 "SSA_RECV" ,"SSA_APPLYRV",
                                 "EIP_YN" ,"EXPNS_DIF","HADCOVID",
                                 "GETVACRV"))

#REDEFINED TARGET VARIABLE
df$GETVACRV[df$GETVACRV == 2] <- 1 #def get vac
df$GETVACRV[df$GETVACRV == 3] <- 0 # not get vac
df$GETVACRV[df$GETVACRV == 4] <- 0


df$GETVACRV <- factor (df$GETVACRV)
df <- na.omit(df)
nums <- c("NUMADLT",'TNUM_PS','THHLD_NUMKID','THHLD_NUMADLT','AGE','TSPNDPRPD',
          'TSPNDFOOD')
noms <- names(df)[!names(df) %in% c(nums, "GETVACC")]
## Convert Nominal Variables
df[ ,noms] <- lapply(X = df[ ,noms], 
                     FUN = factor)
summary(df)
vars <- c(noms, nums)
anyNA(df)

summary(df$GETVACRV )

#GENERATING ANALYSIS AND PLOT FOR VACCINATION INTENTION

plot(df$GETVACRV  ,
     main = "Vaccination Intention", col = c("red","steelblue"),
     ylab = "Count", pch = 19, legend = TRUE)
legend("topleft", legend = c("Would NOT get Vaccine", "Would get Vaccine"),
       lwd = 2, col = c("red", "steelblue"))

#Vaccination Intention based on gender
ggplot(data = df, aes(x = GETVACRV, fill = EGENDER)) +
  geom_bar(position = "dodge")+
  labs(x = "Gender", y = "Count", title = "Vaccination Intention Based on Gender") +
  scale_fill_discrete(labels = c("Male", "Female"))+
  scale_x_discrete(labels=c("0" = "Would Get Vaccine", "1" = "Would NOT Get Vaccine"
                            ))+
  theme_minimal()

#
df$AgeBin <- cut (df$AGE, breaks = c (0,20,30,40,50,60,70,80,100),
                             labels = c("18-20",'20-30','30-40','40-50','50-60','60-70','70-80','80 and older'))
#REDEFINE EDDUCATION VARIABLE
df$EEDUC[df$EEDUC == 2] <- 1 #hs
df$EEDUC[df$EEDUC == 3] <- 1 #hs
df$EEDUC[df$EEDUC == 4] <- 3 #college
df$EEDUC[df$EEDUC == 5] <- 3#college
df$EEDUC[df$EEDUC == 6] <- 5# BS or higher
df$EEDUC[df$EEDUC == 7] <- 5# BS or higher
df$EEDUC <- factor(df$EEDUC)

ggplot(data = df, aes(x = EEDUC, fill =  GETVACRV)) +
  geom_bar(position = "dodge")+
  labs(x = "EDUCATION LEVEL", 
       y = "Count", title = "Vaccination Intention based on Education") +
    scale_x_discrete( labels = c("High School or Less",  "Some College Level","Bachelor+Graduate Level"))+
   scale_fill_discrete(labels = c("Would Not Get Vaccine", "Would Get Vaccine")
   )+
  theme_minimal()

table(df$GETVACRV, df$EEDUC) # GET RESULTS OF VACCINATION INTENTION AND EDDUCATION




set.seed(232)
samp <- createDataPartition(df$GETVACRV, p=.7, list=FALSE) #70/30 SPLIT TEST AND TRAIN
train = df[samp, ] 
summary(train$GETVACRV)
test = df[-samp, ]
summary(test$df$GETVACRV)


### Resampling Methods

### Default, Base Model (with class imbalance)

ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 3)


#  randomly sampling
set.seed(232)
train_ds <- downSample(x = train, 
                       y = train$GETVACRV, # target
                       yname = "GETVACRV")
summary(train_ds)

par(mfrow = c(1,2))
plot(train$GETVACRV, main = "Original")
plot(train_ds$GETVACRV, main = "DownSample")
par(mfrow = c(1,1))

#  Hyperparameter Tuning Model
#perform a grid search for the optimal cp value.


grids <- expand.grid(cp = seq(from = 0,
                              to = 0.05,
                              by = 0.0005))
grids

# choose the cp that is associated with the smallest 
# cross-validated error (highest accuracy)

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     search = "grid") # Cross validation using repeated method

set.seed(232)


DTFit <- train(form = GETVACRV ~ ., 
               data = train, 
               method = "rpart", 
               trControl = ctrl, 
               tuneGrid = grids) # train the DT model using 10-Fold Cross 
# Validation (repeated 3 times).


# cross validation
DTFit

# cp value vs. Accuracy
plot(DTFit)

cc.rpart <- rpart(formula = GETVACRV ~ ., 
                  data = train_ds, 
                  method = "class",
                  control = rpart.control(minsplit=30, cp=0.005))

cc.rpart #output



round((cc.rpart$variable.importance),2)
# obtain variable importance
round(cbind(Training = DT_train_conf$overall,
            Testing = DT_test_conf$overall),2)

barplot(cc.rpart$variable.importance, xlab="variable", 
        ylab="Importance", xaxt = "n", pch=20)
axis(1, at=1:14, labels=row.names(cc.rpart))



rpart.plot(cc.rpart, extra = 3)  ## Tree Plots


   ## Training Performance generate class predictions for our training set
base.trpreds <- predict(object = cc.rpart, 
                        newdata = train_ds,
                        type = "class")
#summary(cc.rpart)
confusionMatrix(base.trpreds, as.factor(train_ds$GETVACRV))

# Obtain a confusion matrix and obtain performance measures for our model applied to the training dataset (train).
DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = as.factor(train_ds$GETVACRV), # actual
                                 positive = "1",
                                 mode = "everything")
DT_train_conf

## Testing Performance generate class predictions for our testing set
base.testpreds <- predict(object = cc.rpart, 
                          newdata = test, 
                          type = "class")


# Obtain a confusion matrix and obtain performance 
# measures for our model applied to the testing dataset (test).

DT_test_conf <- confusionMatrix(data = base.testpreds, # predictions
                                reference = as.factor(test$GETVACRV), # actual
                                positive = "1",
                                mode = "everything")
DT_train_conf

## Goodness of Fit
# Overall
round(cbind(Training = DT_train_conf$overall,
            Testing = DT_test_conf$overall),2)




