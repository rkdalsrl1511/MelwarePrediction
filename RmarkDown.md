머신?��?�� 조별 ?��로젝?��
================
?��준?��-?��?��?��-김?��?��-강�?�기
2019?�� 2?�� 26?��

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

?��?�� ?��?��지 불러?���? & ?��?��공간 ?��?��?���?
----------------------------------------

``` {r}
library(tidyverse)
library(dplyr)
library(randomForest) # ?��?��?��?��?��?��
library(rpart) # ?��?��결정?���? 
library(caret) # ?��?��?��?��?�� ?��?��?�� ?��?��지
library(MLmetrics) # F1 ?��?��?�� ?��?��?�� ?��?��지
library(purrr)
library(e1071)
library(xgboost) # xgboost
library(ROCR) # roc 커브?�� ?��?��?�� ?��?��지
library(pROC) # auroc

setwd('d:/fastcampus/') # ?��?��공간 ?��?��?���?
getwd() # ?��?��?�� ?��?��공간 ?��?��?���?
```

``` {r}
# ?��벨을 빈도?�� ?��?�� 50개로 ?��축하?�� ?��?��
# ?��?�� : 목표변?��(trainset.q$HasDetections), ?��?��변?��(trainset.q[,factor.name[i]])
CompressLevels <- function(object, input, Nlevel = 50){
    
    # factor?�� ?��벨에 ?���? 목표변?��?�� 0�? 1?�� 빈도�? 지?��변?��?�� ?��?��
    detector <- by(object,
                   input,
                   table)
    
    # ?���? 축소�? ?��?��?�� ?��?��?��?���? character�? 변?��
    input <- as.character(input)
    
    # ?��벨의 ?��가 50�? ?��?��?�� 경우
    if(length(detector) > Nlevel){
      
      # ?��벨에 ?���? 백분?��?�� ?��?�� ?��로운 객체
      detector.prop.vector <- c()
      
      # �? ?��벨에 ?���? 빈도�? 백분?���? ?��?��?���?
      for (k in 1:length(detector)) {
        
        detector.prop <- 100 * detector[[k]][1] / (detector[[k]][1] + detector[[k]][2])
        
        detector.prop.vector <- rbind(detector.prop.vector, detector.prop)
        
      }
      
      # cut?�� ?��?��?�� factor�? ?��?��?���?
      detector.prop.factor <- cut(detector.prop.vector,
                                  breaks = seq(from = 0,
                                               to = 100,
                                               by = 100 / Nlevel),
                                  right = FALSE)
      
      # cut?�� ?��?�� 변?��?�� factor�? ?��?��?���?
      for(k in 1:length(detector)){
        
        # �? ?��벨에 ?��?��?��?�� ?��?��?�� ?��부 �? ?���? 값으�? 바꾸�?
        
        # 백분?��?�� 100?��?��?��, ?��?�� ?��벨이 NA값인 경우
        if(is.na(detector.prop.factor[k])){
          
          input[input == names(detector)[k]] <- '[all]'
          
          # �? ?��
        }else{
          
          input[input == names(detector)[k]] <- as.character(detector.prop.factor[k])
          
        }
        
      }
      
    }
  
    # ?��?�� factor�? 변?��
    input <- as.factor(input)
    
    return(input)
    
}

# ?��벨의 ?��름을 ?��?��?�� 만큼 ?��?��주는 ?��?��
CutLevels <- function(data.variable, start, end){
  
  data.variable <- data.variable %>% as.character()
  data.variable <- data.variable %>% str_sub(start = start, 
                                             end = end)
  data.variable <- data.variable %>% as.factor()
  
  return(data.variable)
  
}

# NA�? '미응?��'(default)?���? 변?��?���?, factor�? 변?��?��주는 ?��?��
NAtoFactor <- function(data.variable, NA.message = '미응?��'){
  
  data.variable <- as.character(data.variable)
  
  data.variable[is.na(data.variable) == TRUE] <- NA.message
  
  data.variable <- as.factor(data.variable)
  
  return(data.variable)
  
}
```

?��?��?��?�� 불러?���?
=================

``` {r}
dataset <- read.csv(file = 'trainset_mini.csv',
                    header = TRUE)

# ?���? ?��거하�?
dataset <- dataset[,-1]

# HasDetections : 목표변?��. factor�? 변?��
dataset$HasDetections <- as.factor(dataset$HasDetections)
dataset$HasDetections <- relevel(dataset$HasDetections, ref = '1')
```

1. DataSet<br><br><br>
----------------------

<div style = "color:red">
1.  ?��?��?��?�� 출처
    </div>
    <https://www.kaggle.com/c/microsoft-malware-prediction>

<br>Kaggle Research Prediction Competition<br>( kaggle?��?�� 주�?�?��?�� ?���? ??�?��)<br> ?��?�� ?��?��?��?�� ?�� 1500만행, 83개의 column?�� 가지�? ?��?��.<br> �? 중에?�� 800만행??� 목표변?���? ?��?��?��?�� trainset?���? ?��공하�? ?��?���?, ?��머�?� 700만행?��?�� 목표변?���? ?��?��?��?�� testset?���? ?��공하�? ?��?��.<br><br>목표변?��?�� HasDetections?��?�� 컬럼?��로서, 1�? 0?���? ?��루어?�� ?���? ?��문에 ?��리들?�� 목적??� ?��진분류�?? ?��?�� ?��?��코드 감염 ?��부 ?��측이?���? ?�� ?�� ?��겠다.<br><br> ?��리조?�� �? 조원?��?�� 컴퓨?�� ?��건을 고려?��?�� trainset ?��?��?�� 중에?�� **1%만을 sample?��?���? 추출?��?�� ?��것을 ?��?�� 0.7:0.3?�� 비율�? trainset�? validationset**?���? ?��?��?��?�� ?��측에 ??�?�� 지?��?��?�� ?��?��?��기로 ?��??�?��.<br><br><br><br>
<div style = "color:red">
1.  ?��?��?��?�� 구조
    </div>
    **<?��?��?�� 변?�� ?���?.hwp 참고>** �? column?�� ?���? �? NA로서 마이?��로스?��?��?��?�� ?��별한 주석?�� ?��공하지 ?��??� 변?�� 20개�?? ?��?��?��?�� ?��?��?���?(?��?��?��7~10)�? 각종 Identifier?��?�� 존재?��?��.<br>

**str?�� ?��?�� 간단?���? ?��?��?���? ?��?��?��?��**

``` {r}
str(dataset) # ?��?��?��?��?�� 구조
```

<br><br> **1%�? ?��?��링한 ?��?��?��?��?�� 길이**

``` {r}
nrow(dataset) # 89155�?
```

<br><br> **1%�? ?��?��링한 ?��?��?��?��?�� 목표변?��?�� 1�? 0?�� 비율**

``` {r}
dataset$HasDetections %>% table() %>% prop.table()
```

<br><br> **�? 변?���? NA�? ?��?��?���?**

``` {r}
sapply(dataset, function(x) sum(is.na(x)))
```

<br><br> ?��처리 ?��기에 ?��?��?��, NA가 ?��?��?�� ?��?��?���? 처리?���? ?�� ?��?�� 모형?�� ?���?, NA가 ?��?��?���? ?��?�� 모형?�� ?��?��.<br>?��?��가지 모형?��?�� ?��?��?�� ?��가?���? ?��?��?��?�� NA?�� ?��?��게든 처리?��주는 ?��?�� 좋을 �? 같다.<br> ?��?���? ?��?�� IT기업?�� ?��비자?��?�� ?��?��코드 감염 ?��부�? ?��측할 ?��, 모든 ?��?��?���? ?��부 조사?��기는 ?��?�� 것이?��.<br> 컴퓨?��?�� ?��가견이 ?��?�� ?��?��?��?�� ?��?��?�� ??�?��?��?�� ?��?��?��?�� ?��?��?�� 컴퓨?��?�� ??�?��?�� ?�� ?��지 못하�?, ?��?��?��?�� ?��부�? 건들?��기�?? 꺼려?��?��. ?��?��, ?��?���? ?��?��?��?��?�� ??�?��?���? ?��?��코드?�� ??�?�� ?��측을 ?��고자?�� ?��, **기업?�� 미쳐 ?��?��?��지 못한 것들?�� ?��?�� 것이?��. ?��것들?�� 모두 고려?��?�� 최�?�?�� ?��?��?�� ?��측을 ?��?�� 모형?�� 기업?�� ?��?��?�� 모형?�� 것이?��.** ?��?��?�� NA?���? ?��?��?��버리?�� 것�?� ?�� 좋�?� ?��?��?��?��?�� ?��?��.<br><br> ?��리조?��?�� ?��각해�? NA ?��처리문제 ?��결법<br> 1. 보류?��?��.<br> 2. 모두 ?��거한?��.<br> **3. ?�� 3?�� 범주�? 만든?��. ( 범주?��?���? 만들?��?�� ?��결하�? )**<br> **4. ??�체값?�� 찾는?��. (?��, int?��?���? 만들 ?�� ?��?�� 변?��?���?)**<br> 5. 기�?� 방법<br><br><br> ?��?��?�� 말했?��?��, 1번의 NA�? ?��?��?�� 보류?��?�� 것�?� ?��?��?�� ?��비일 ?��?�� ?��?��. 그리�? 2번의 NA�? 모두 ?��거하?�� 것�?� ?��?��?�� NA�? 보류?��?�� 것과 같�?� 말이?��. 그리�? ?��리�?� ?���? 고려?��?��?�� 방법??� 3번과 4번일 것이?��.<br> 5번의 경우?��, ?��?�� 모형?�� 만들?��보고, �? 변?��?�� �? 중요?��가 ?��??� 것들?�� 중심?���? ?��처리?��?�� 방식 ?��?�� ?��?�� 것이?��. ?�� 방식?��??� ?�� ?��로젝?��가 ?��?�� ?��?�� 개인?��?���? 만들?���? ?��각이?��.<br><br><br><br>

2. ?��처리 ?���? ?��?�� ?��?��?��?�� ?��?��결정?���?<br>
---------------------------------------------

?��처리�? ?��지 ?��??� ?��?��?��?�� 만든 모형??� ?��마도 NA�? 모두 ?��거한 ?��?��??� 같을 것이?��. ?��?��?�� ?��결법 1번과 2번에 ?��?��?��?�� 방식?�� 것이?��.<br><br>

\*\* dataset?�� trainset�? validationset?���? ?��?���?\*\*

``` {r}
set.seed(123)

index <- sample(1:2,
                size = nrow(dataset),
                prob = c(0.7,0.3),
                replace = TRUE)

# t??� trainset, v?�� validationset?��?��.
# ?��?�� testset?�� 목표변?���? ?�� ?�� ?��?��므�?, ?���? ?�� ?��?�� dataset?�� q1�? q2�? 분리?��?�� ?��측률?�� ?��?��?��?���? ?��?��.
dataset.t <- dataset[index == 1, ]
dataset.v <- dataset[index == 2, ]
```

<br><br> \*\* ?��?��결정?���? 모형 ?��?��?��보기\*\*

``` {r}
fitTree <- rpart(HasDetections ~.,
                 data = dataset.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

trPred <- predict(fitTree,
                  newdata = dataset.v,
                  type = 'class')

trReal <- dataset.v$HasDetections

confusionMatrix(trPred, trReal, positive = '1')
F1_Score(trPred, trReal)
```

<br><br> **?��?��?��?��?��?�� 모형 ?��?��?��보기**

``` {r}
# AvSigVersion
# AppVersion
# OsBuildLab
# Census_OSVersion

# ?��?��방편?���? ?��벨의 ?��가 많�?� column?�� ?��거하�? ?��?��?��?��?��?�� 모형?�� ?��?��?��??�?��.

dataset.t.i <- dplyr::select(dataset.t,-c(AvSigVersion,
                                          AppVersion,
                                          OsBuildLab,
                                          Census_OSVersion))

dataset.v.i <- dplyr::select(dataset.v,-c(AvSigVersion,
                                          AppVersion,
                                          OsBuildLab,
                                          Census_OSVersion))

fitRFC <- randomForest(x = dataset.t.i[complete.cases(dataset.t.i),
                                       -78],
                       y = dataset.t.i[complete.cases(dataset.t.i),
                                       78],
                       ntree = 100,
                       mtry = 10,
                       importance = TRUE,
                       do.trace = 50,
                       keep.forest = TRUE)


trPred <- predict(fitRFC, dataset.v.i[complete.cases(dataset.v.i),],
                  type = 'response')
trReal <- dataset.v.i[complete.cases(dataset.v.i),78]

confusionMatrix(trPred, trReal, positive = '1')
F1_Score(trPred, trReal)
```

<br><br> ?��?��결정?���? 보다?�� ?��?��?�� ?��??� 민감?���? 보이�?, NA값들?�� ?��거하�? 보니, ?��??� ?��?��?�� 거의 ?��?��?��?�� ?��??�?��. ?��?��?�� ?��처리�? ?��?��?��, NA값을 범주�? 만들거나, ?��?�� 변?��?��?�� int?��?���? 변?��?�� ?��, NA값을 ??�체값?���? ??�체하?�� 방법?�� ?��?��?��기로 ?��??�?��.<br><br><br><br>

3. NA값이 ?��?�� 변?��?��?�� 범주?��?���? ?��처리?���?
---------------------------------------------

<br><br> ?��?��?�� 보았?��?��, NA값을 ?��처리?��지 ?���?, 그�?��? ?��?��?��?���? 좋�?� ?���? 모형?�� 기�?�?���? ?��?��?�� �? 같다. 그냥 50%?��률로 찍어?�� ?��측하?�� 것과 비슷?�� ?��?��?��?��.<br>**?��?��, ?��?��?���? ?��?��?��?�� 과정?��?�� NA값에 ?��미�?� ?��?�� 경우�? ?��?��?��??�?��.** ?���? ?��?���?, 변?�� **IsProtected**?�� 경우?��?�� 1?�� ?��?�� 백신?�� ?��?�� �?, 0?�� ?��?�� ?��?��?��?���? ?��지 ?��??� 백신?�� ?��?�� �?, **NA?�� 경우 백신?�� ?��?��?��지 ?��?��?��.** ?��?�� ?��미�?� ?��?��.<br><br>?��?��?�� NA�? 범주�? 처리?��보기�? ?��??�?��.<br><br>

``` {r}
# ?��처리?�� ?��?��?��?��
dataset.q <- dataset
```

**버전?�� ?���? ?��?�� factor**

``` {r}
# 버전?�� ?���? ?��?�� factor
factor.name <- c('EngineVersion',
                 'AppVersion',
                 'AvSigVersion',
                 'Census_OSVersion')

factor.cutnum <- c(6,7,4,7)

for(i in 1:4){
  
  dataset.q[,factor.name[i]] <- CutLevels(dataset.q[,factor.name[i]],
                                           start = 1,
                                           end = factor.cutnum[i])
  
}
```

<br><br> **""?��?�� ?��름의 ?��벨을 가�? factor?�� ??�?��?�� '미응?��'?���? ?���? 바꾸�?**

``` {r}
# "" 가 ?��?��?�� factor�? '미응?��'?���? 바꾸�?
factor.name <- c('Census_PrimaryDiskTypeName',
                 'Census_ChassisTypeName',
                 'Census_PowerPlatformRoleName')

for(i in 1:3){
  
  dataset.q[,factor.name[i]] <- as.character(dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- ifelse(dataset.q[,factor.name[i]] == "", yes = "미응?��", dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- as.factor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **NA값이 ??�?��?��?�� 변?��**

``` {r}
# NA값이 ??�?��?��?�� 변?��
factor.name <- c('DefaultBrowsersIdentifier',
                 'OrganizationIdentifier',
                 'Census_IsFlightingInternal',
                 'Census_ThresholdOptIn')

for(i in 1:length(factor.name)){
  
  dataset.q[,factor.name[i]] <- NAtoFactor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **기�?� 변?��**

``` {r}
# 기�?� 변?��
factor.name <- c('IsBeta',
                 'IsSxsPassiveMode',
                 'AVProductStatesIdentifier',
                 'HasTpm',
                 'CountryIdentifier',
                 'CityIdentifier',
                 'GeoNameIdentifier',
                 'LocaleEnglishNameIdentifier',
                 'OsSuite',
                 'OsBuild',
                 'IsProtected',
                 'Census_HasOpticalDiskDrive',
                 'Census_OSBuildNumber',
                 'Census_OSBuildRevision',
                 'Census_OSInstallLanguageIdentifier',
                 'Census_OSUILocaleIdentifier',
                 'Census_IsPortableOperatingSystem',
                 'Census_IsFlightsDisabled',
                 'Census_FirmwareManufacturerIdentifier',
                 'Census_FirmwareVersionIdentifier',
                 'Census_IsSecureBootEnabled',
                 'Census_IsWIMBootEnabled',
                 'Census_IsVirtualDevice',
                 'Census_IsTouchEnabled',
                 'Census_IsPenCapable',
                 'Census_IsAlwaysOnAlwaysConnectedCapable',
                 'Wdft_IsGamer',
                 'Wdft_RegionIdentifier',
                 'AutoSampleOptIn',
                 'SMode',
                 'IeVerIdentifier',
                 'Firewall',
                 'UacLuaenable',
                 'Census_OEMNameIdentifier',
                 'Census_OEMModelIdentifier',
                 'Census_ProcessorManufacturerIdentifier',
                 'RtpStateBitfield',
                 'AVProductsInstalled',
                 'AVProductsEnabled',
                 'Census_ProcessorModelIdentifier',
                 'Census_InternalBatteryNumberOfCharges')

for(i in 1:length(factor.name)){
  
  dataset.q[,factor.name[i]] <- NAtoFactor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **?��치형, ?��??� 범주?��?���? ?��?��?�� ?�� ?��?�� 변?��**<br> ?�� 변?��?��?�� 경우?��?�� 2가지 모두 ?��?��?��?�� ?��?��?��본다.

``` {r}
# int?��?���? 그�?��? ?��?��?�� ?�� ?���?, ?��??� 범주?��?���? ?��?��?�� ?�� ?��?�� 변?��
# ?�� 변?��?�� 경우 2가지�? 모두?��?�� ?��?��?��본다.

# 범주?��?���? ?��?��?�� ?��?��?��?��
dataset.q.1 <- dataset.q
# int?�� 그�?��? ?��?��?�� ?��?��?��?��
dataset.q.2 <- dataset.q


factor.name <- c('Census_ProcessorCoreCount',
                 'Census_PrimaryDiskTotalCapacity',
                 'Census_SystemVolumeTotalCapacity',
                 'Census_TotalPhysicalRAM',
                 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                 'Census_InternalPrimaryDisplayResolutionHorizontal',
                 'Census_InternalPrimaryDisplayResolutionVertical')


# (1) NA처리?���?, 범주?��?���? 변?��?���?
for(i in 1:length(factor.name)){
  
  dataset.q.1[,factor.name[i]] <- NAtoFactor(dataset.q.1[,factor.name[i]])
  
}

# (2) NA�? ??�체값?���? ??�체하�?
for(i in 1:length(factor.name)){

  dataset.q.2[is.na(dataset.q.2[,factor.name[i]]),
              factor.name[i]] <- summary(dataset.q.2[,factor.name[i]])[3]

}
```

<br><br> **?��처리?�� ?��?��?��?�� 1번의 NA??� ?��벨수 ?��?��?���?**

``` {r}
# NA값과 ?��벨수 ?��?��?���?
factor.name <- colnames(dataset.q.1)

for(i in 1:length(factor.name)){
  
  cat('변?���? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('?��벨수 : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

<br><br> **?��처리?�� ?��?��?��?�� 2번의 NA??� ?��벨수 ?��?��?���?**

``` {r}
# NA값과 ?��벨수 ?��?��?���?
factor.name <- colnames(dataset.q.2)

for(i in 1:length(factor.name)){
  
  cat('변?���? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('?��벨수 : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

4.?��처리?�� ?��?��?��?��?���? ?��?��결정?���? ?��?��?���?<br><br>
-----------------------------------------------------

**?��처리?�� ?��?��?��?��?���? ?��?��결정?���? ?��?��?���? 1**

``` {r}
dataset.q.1.t <- dataset.q.1[index == 1, ]
dataset.q.1.v <- dataset.q.1[index == 2, ]

fitTree <- rpart(HasDetections ~.,
                 data = dataset.q.1.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

trPred <- predict(fitTree,
                  newdata = dataset.q.1.v,
                  type = 'class')

trReal <- dataset.q.1.v$HasDetections

confusionMatrix(trPred, trReal, positive = '1')
F1_Score(trPred, trReal)
```

<br><br> **?��처리?�� ?��?��?��?��?���? ?��?��결정?���? ?��?��?���? 2**

``` {r}
dataset.q.2.t <- dataset.q.2[index == 1, ]
dataset.q.2.v <- dataset.q.2[index == 2, ]

fitTree <- rpart(HasDetections ~.,
                 data = dataset.q.2.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

trPred <- predict(fitTree,
                  newdata = dataset.q.2.v,
                  type = 'class')

trReal <- dataset.q.2.v$HasDetections

confusionMatrix(trPred, trReal, positive = '1')
F1_Score(trPred, trReal)
```

5. 변?�� ?���? 축소?��보기
-----------------------

?��부 변?��?��?�� NA값을 Mean?���? ??�체할 ?��?�� ?��?��?��가 조금 ?�� ?��?��?��. ?��지�?, ?��무런 ?��처리�? ?��지 ?��??� ?��?��?��?�� ?��?��?�� ?��?��결정?���? 모형보다?�� 부족한 ?��치�?? 보이�? ?��?��. identifier??� 같�?� 변?��?��?�� factor�? 변?��?��면서 ?��많�?� ?��벨이 ?��?��?��?�� ?��?��?�� ?��체적?�� 지?��?��?�� ?��치�?� 감소?�� �? 같다.<br><br>?��?��?�� ?��벨의 ?���? ?��?��?��?�� 줄인?���?, ?�� 좋�?� 결과가 ?��?�� 것이?���? ?��?��?��?��.<br><br>HasDetections(목표변?��)?�� 1�? 0?�� 빈도가 ?��?��?�� 것끼�? 그룹?���? 묶어?�� ?��벨의 ?��가 50�? ?��?��?�� 컬럼?�� 50�? ?��?��?�� ?��벨로 줄여보도�? ?��??�?��.<br><br> **50�? ?��?��?�� ?��벨의 ?�� 줄이�?**

``` {r}
factor.name <- colnames(dataset.q.1)
Com.factor.name <- c() # ?��벨이 50�? ?��?��?�� 컬럼�?


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.1[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.1)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.1[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.1$HasDetections, input = dataset.q.1[,Com.factor.name[i]])
  
}

# ?���? ?�� ?��?�� ?�� �? ?��?��?��보기
for(i in 1:length(factor.name)){
  
  cat('변?���? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('?��벨수 : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

``` {r}
factor.name <- colnames(dataset.q.2)
Com.factor.name <- c() # ?��벨이 50�? ?��?��?�� 컬럼�?


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.2[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.2)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.2[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.2$HasDetections, input = dataset.q.2[,Com.factor.name[i]])
  
}

# ?���? ?�� ?��?�� ?�� �? ?��?��?��보기
for(i in 1:length(factor.name)){
  
  cat('변?���? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('?��벨수 : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

6. 축소?�� ?��?��?��?��?���? ?��?��결정?���?, ?��?��?��?��?��?�� ?��?��?���?
----------------------------------------------------------

<br><br> ?��?��부?�� ?��?��결정?��무�?? ?��?��?��보고, ?��?��?��?��?��?��?�� ?��?��까�?� ?��보겠?��.<br><br><br><br> **?��?��결정?���? ?��?��?���? 1**

``` {r}
dataset.q.1.t <- dataset.q.1[index == 1, ]
dataset.q.1.v <- dataset.q.1[index == 2, ]

fitTree <- rpart(HasDetections ~.,
                 data = dataset.q.1.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

trPred <- predict(fitTree,
                  newdata = dataset.q.1.v,
                  type = 'class')

trReal <- dataset.q.1.v$HasDetections

# ?��?��?��?��
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?��?��?��
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# 비용복잡?�� ?�� 출력
printcp(fitTree)
```

**?��?��결정?���? ?��?��?���? 2**

``` {r}
dataset.q.2.t <- dataset.q.2[index == 1, ]
dataset.q.2.v <- dataset.q.2[index == 2, ]

fitTree <- rpart(HasDetections ~.,
                 data = dataset.q.2.t,
                 method = 'class',
                 parms = list(split = 'gini'),
                 control = rpart.control(minsplit = 20,
                                         cp = 0.01,
                                         maxdepth = 10))

trPred <- predict(fitTree,
                  newdata = dataset.q.2.v,
                  type = 'class')

trReal <- dataset.q.2.v$HasDetections

# ?��?��?��?��
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?��?��?��
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# 비용복잡?�� ?�� 출력
printcp(fitTree)
```

<br><br><br> ?��벨의 ?���? 축소?�� ?��?��로는 모든 컬럼?�� 범주?��?���? 변경한 ?��?��?��?��?��?�� ?�� ?��??� F1 ?��?��??� auroc 값이 ?��?��?��. ?��?��?�� ?��?��?��?��?��. 각각 모형?��?�� 가?�� 마�?��? 분리?��?�� xerror가 가?�� ?���? ?��?��?��므�?, 가지치기�? ?�� ?��?��가 ?��?��?��. 그렇?���? ?��?�� 모든 컬럼?�� 범주?��?���? 변경한 ?��?��?��?��?���? ?��?��?��?��?��?�� 모형?�� ?��?��?��보기�? ?��??�?��.<br><br> **?��?��?��?��?��?�� 모형 간단?���? ?��?��?��보기**

``` {r}
fitRFC <- randomForest(x = dataset.q.1.t[,-82],
                       y = dataset.q.1.t[, 82],
                       ntree = 100,
                       mtry = 10,
                       importance = TRUE,
                       do.trace = 50,
                       keep.forest = TRUE)


trPred <- predict(fitRFC, 
                  dataset.q.1.v,
                  type = 'response')
trReal <- dataset.q.1.v$HasDetections

# 모형 ?��?�� 결과 ?��?��?���? ( ?��분류?�� ?��?��?���? )
print(fitRFC)

# 변?�� 중요?�� 출력?���?
importance(fitRFC)

# 마진 그래?��
plot(margin(fitRFC))

# ?��?��?��?��
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?��?��?��
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> MeanDecreaseAccuracy?��?�� ?���? ?��?��?�� 주는 변?��?��,<br> AppVersion 13.2078028<br> AVProductStatesIdentifier 21.1545553<br> CountryIdentifier 10.5903571<br> CityIdentifier 51.7872658<br> SmartScreen 31.7732623<br> Census\_OEMModelIdentifier 40.9204674<br> Census\_ProcessorModelIdentifier 13.0899241<br> Census\_SystemVolumeTotalCapacity 103.9377516<br> Census\_InternalBatteryNumberOfCharges 16.4877933<br> Census\_OSInstallLanguageIdentifier 13.7160384<br> Census\_FirmwareVersionIdentifier 31.2409193<br> Wdft\_RegionIdentifier 12.5699366<br><br> ?��것들?��?��. �? 변?��?��?�� ?��미�?? 간단?���? ?��?��보자�?,<br> 1. windows defender?�� 버전<br> 2. ?��?�� 바이?��?�� 백신?�� 버전<br> 3. �?가코드, ?��?��코드, 지?��코드, ?��?��코드<br> 4. smartscreen(?��?��?�� 10?�� 방화�? 기능)<br> 5. OEM?�� 종류(?��?��체제�? ??�?��?���? 구매?��?�� ?��치하?�� 방식)<br> 6. CPU?�� 모델 �?<br> 7. ?��?��체제가 깔려?��?�� ?��?��?��?�� ?���?<br> 8. 방화벽의 버전<br> ?��?��. 그리�? ?��?��?��?��?�� ?��?��?���? 결과, ?�� 모형??� 감염?��지 ?��??� PC�? ?�� ?�� 찾아?��?�� ?��?��?��??� ?��밀?��가 ?��??� 모형?��?��.<br> ?�� 모형??� ?��밀?��가 ?��?��?�� ?��??� ?��?��지�?, ?��?�� 코드?�� 감염?�� 컴퓨?���? 감염?��지 ?��?��?���? ?��?��?��?�� 경우가 ?��?��?���? ?��문에 ?��?��?�� 모형?��?��.<br> 그렇�? ?��문에 ?��?��?�� ?��?���?, 민감?��가 ?�� ?��??� 모형?�� ?��?�� ?�� ?��지 ?��?���? ?��각을 ?��보았?��.<br> **?��?��?��?��?��?�� 모형 ?��?��?��보기**

``` {r}
# # �? 32개의 조합
# # ??� ?��?�� 좋�?� ?��?��가 ?��?��?��?
# grid <- expand.grid(ntree = c(100),
#                     mtry = c(3,5,7,8,12,13,14,15))
# 
# tuned <- data.frame()
# 
# for (i in 1:nrow(grid)){
#   
#   set.seed(123)
#   
#   cat('ntree : ', grid[i,'ntree'],
#       'mtry : ', grid[i, 'mtry'],'\n')
#   
#   fit <- randomForest(x = dataset.q.1.t[,-82],
#                       y = dataset.q.1.t[, 82],
#                       xtest = dataset.q.1.v[,-82],
#                       ytest = dataset.q.1.v[, 82],
#                       ntree = grid[i,'ntree'],
#                       mtry = grid[i,'mtry'],
#                       importance = TRUE,
#                       do.trace = 50,
#                       keep.forest = TRUE)
#   
#   # ?��측값
#   trPred <- fit$test$predicted
#   # ?��?���?
#   trReal <- dataset.q.1.v$HasDetections
#   # ?��?��?��?��
#   con <- confusionMatrix(trPred, trReal, positive = '1')
#   
#   # ?��분류?��
#   mcSum <- sum(fit$predicted != dataset.q.1.t$HasDetections)
#   # ?��분류?��
#   mcrate <- mcSum / nrow(dataset.q.1.t)
#   
#   tuned <- rbind(tuned, 
#                  data.frame(Index = i,
#                             mcRate = mcrate,
#                             sensitivity = con$byClass[1],
#                             specificity = con$byClass[2],
#                             PredValue = con$byClass[3]))
#   
# }
# 
# View(tuned)
```

<br><br> ?��간이 ?��?�� 관계로 매우 간단?���? 그리?�� ?��치�?? ?��??�?��.<br> ntree = 500, mtry = 13?��?���? ?��?��?��?��?��?�� 모형?�� ?��?��?���? ?��?��?�� �? 같다.<br><br> **?��?��?�� ?��?��?��?��?��?�� 모형**

``` {r}
fitRFC <- randomForest(x = dataset.q.1.t[,-82],
                       y = dataset.q.1.t[, 82],
                       ntree = 500,
                       mtry = 13,
                       importance = TRUE,
                       do.trace = 50,
                       keep.forest = TRUE)


trPred <- predict(fitRFC, 
                  dataset.q.1.v,
                  type = 'response')
trReal <- dataset.q.1.v$HasDetections

# 모형 ?��?�� 결과 ?��?��?���? ( ?��분류?�� ?��?��?���? )
print(fitRFC)

# 변?�� 중요?�� 출력?���?
importance(fitRFC)

# 마진 그래?��
plot(margin(fitRFC))

# ?��?��?��?��
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?��?��?��
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

7. 번외 - XGBoost ?��?��?��보기<br><br>
------------------------------------

각종 ??�?��?��?�� ?��??� ?��?���? 기록?��?�� 머신?��?�� ?��고리�? �? ?��?��?��, XGBoost�? ?��?��?��?�� ?��?��?�� ?��가?��보기�? ?��??�?��.<br> **?��벨링, ?��미�?�?�� ?��?��**

``` {r}
# ?��벨링 ?��?�� ( numeric vector�? 반환 )
MakingLabel <- function(data.variable){
  
  for(i in 1:nlevels(data.variable)){
  
    levels(data.variable)[i] <- i
  
  }
  
  data.variable <- as.numeric(data.variable)
  return(data.variable)
}

# ?��미�?�?�� 만드?�� ?��?�� ( ?��미�?�?��?��?�� data.frame?���? 반환 )
MakingDummy <- function(data.variable, data.name){
  
  result <- data.frame(index = 1:length(data.variable))

  for(i in 1:nlevels(data.variable)){
    
      newdummy <- ifelse(data.variable ==
                             levels(data.variable)[i],
                           1,
                           0)
      result <- cbind(result, newdummy)
      newname <- str_c(data.name, i, sep = '.')
      colnames(result)[i+1] <- newname
  
    }

  result <- result[,-1]
  
  return(result)
  
}

# ?��?��?��?��
example.label <- MakingLabel(dataset.q.1$ProductName)
example.dummy <- MakingDummy(dataset.q.1$ProductName,
                             'result')
```

<br><br> **?��?��?��?�� 변?��?���?**

``` {r}
# ?��?��?��?��
dataset.n <- dataset.q.1

HasDetections <- dataset.n$HasDetections %>% as.numeric()
HasDetections <- ifelse(HasDetections == 2,0,1)

dataset.n.label <- dataset.n[, -82]
dataset.n.dummy <- dataset.n

# dataset.n.dummy?�� ?���? ?�� ?��르기
factor.name <- colnames(dataset.n.dummy)
Com.factor.name <- c() # ?��벨이 50�? ?��?��?�� 컬럼�?


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.n.dummy[,factor.name[i]]) > 10){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.n.dummy)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.n.dummy[,Com.factor.name[i]] <- CompressLevels(object = dataset.n.dummy$HasDetections, 
                                                         input = dataset.n.dummy[,Com.factor.name[i]],
                                                         Nlevel = 10)
  
}

# ?���? ?�� ?��?�� ?��?��?���?
for(i in 1:length(factor.name)){
  
  cat('변?���? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.n.dummy[,factor.name[i]]), '\n')
  cat('?��벨수 : ', nlevels(dataset.n.dummy[,factor.name[i]]), 
      '\n\n')
  
}

# 목표변?���? ?��거하�?
dataset.n.dummy <- dataset.n.dummy[, -82]

# factor.name?��?�� 목표변?�� ?��거하�?
factor.name <- colnames(dataset.n)[-82]

# 1. ?��벨링?���?
for(i in 1:length(factor.name)){
  
  dataset.n.label[, factor.name[i]] <- MakingLabel(dataset.n.label[, factor.name[i]])
  
}

dataset.n.label <- cbind(dataset.n.label, HasDetections)
ncol(dataset.n.label)

# 2. ?��미�?�?��?��
for(i in 1:length(factor.name)){
  
  dataset.n.dummy <- cbind(dataset.n.dummy,
                           MakingDummy(dataset.n.dummy[,factor.name[i]],
                                       factor.name[i]))
  
}

dataset.n.dummy <- dataset.n.dummy[,-c(1:81)]
dataset.n.dummy <- cbind(dataset.n.dummy, HasDetections)
ncol(dataset.n.dummy)

# trainset�? validationset?���? ?��?���?
dataset.n.label.t <- dataset.n.label[index == 1, ]
dataset.n.dummy.t <- dataset.n.dummy[index == 1, ]

dataset.n.label.v <- dataset.n.label[index == 2, ]
dataset.n.dummy.v <- dataset.n.dummy[index == 2, ]
```

<br><br> **XGBoost ?��?��?��보기 1**

``` {r}
# 1. ?��벨링?�� ?��?��?��?��
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.label.t[ ,-82]),
                      label= as.matrix(dataset.n.label.t[ , 82]))


# ?��?��미터
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds 찾기
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost 모형 ?��?��?���?
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.label.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.label.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# ?��?��?��?��
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc ?��?��?��
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> **XGBoost ?��?��?��보기 2**

``` {r}
# 2. ?��미�?�?���? 만든 ?��?��?��?��
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.dummy.t[ ,-82]),
                      label= as.matrix(dataset.n.dummy.t[ , 82]))


# ?��?��미터
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds 찾기
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost 모형 ?��?��?���?
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.dummy.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.dummy.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# ?��?��?��?��
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc ?��?��?��
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

#### ?<br><br>

8. 마무�?
---------

?��리들?�� 관?��분야?�� ?��?��코드?�� ??�?�� ?���? ?��?��?��?��?�� ??�?��?��?�� ?��?��?��?�� 개요가 ?��?�� ?��?��??�지�?, ?��?���? ?��처리?�� ?��?��?�� ?��?��가지 ?��?��?�� ?��계점?��?�� ?��?��?��.<br> 1. 1500만행?�� ?��?�� ?��?��?��?��?��?�� testset?��?�� ?��지�?, trainset?��?�� ?��?�� Identifier가 ?���? ?��문에, ?�� 5~10%?��?��?�� ?��보�?? ?��?��?��지 못하?�� 것을 ?��?��?��?��.<br><br> 2. ?��?�� ?��?��?��?��?��?�� ?��?��?�� 방법 �? 몇군??��? ?��?��?��?��?���?, 그로 ?��?��?�� 치명?��?�� ?��차�?� 발생?�� ?��?�� ?��?��.<br><br> 3. 컴퓨?��?�� ?��?�� 문제�? ?��?�� ?��?��?��?�� ?��처리 ?��간에�? 7?�� ?��?��?�� ?��?��?�� 것이?��.<br><br> 4. 그렇�? ?��문에, ?��?�� ?��?��?���? ?��뤄보지 못하??��?, ?��?��?�� 방법?��?�� 고�?�하기에?�� ?��계�?� ?��?��?��. ?�� 과정?�� ?��?�� ?��?�� 개인?��?���? ?�� ??�?���? 마무�? 지?�� ?��각이?��.
