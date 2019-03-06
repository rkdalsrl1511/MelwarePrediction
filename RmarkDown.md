머신러닝 조별 프로젝트
================
오준승-윤휘영-김수현-강민기
2019년 2월 26일

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

필요 패키지 불러오기 & 작업공간 설정하기
----------------------------------------

``` {r}
library(tidyverse)
library(dplyr)
library(randomForest) # 랜덤포레스트
library(rpart) # 의사결정나무 
library(caret) # 혼동행렬에 필요한 패키지
library(MLmetrics) # F1 점수에 필요한 패키지
library(purrr)
library(e1071)
library(xgboost) # xgboost
library(ROCR) # roc 커브에 필요한 패키지
library(pROC) # auroc
library(knitr)

setwd('d:/fastcampus/') # 작업공간 설정하기
getwd() # 설정된 작업공간 확인하기
knit("RmarkDown.Rmd")
```

``` {r}
# 레벨을 빈도에 따라 50개로 압축하는 함수
# 인자 : 목표변수(trainset.q$HasDetections), 입력변수(trainset.q[,factor.name[i]])
CompressLevels <- function(object, input, Nlevel = 50){
    
    # factor의 레벨에 따른 목표변수의 0과 1의 빈도를 지역변수에 할당
    detector <- by(object,
                   input,
                   table)
    
    # 레벨 축소를 위해서 임시적으로 character로 변환
    input <- as.character(input)
    
    # 레벨의 수가 50개 이상인 경우
    if(length(detector) > Nlevel){
      
      # 레벨에 따른 백분율을 담을 새로운 객체
      detector.prop.vector <- c()
      
      # 각 레벨에 따른 빈도를 백분율로 전환하기
      for (k in 1:length(detector)) {
        
        detector.prop <- 100 * detector[[k]][1] / (detector[[k]][1] + detector[[k]][2])
        
        detector.prop.vector <- rbind(detector.prop.vector, detector.prop)
        
      }
      
      # cut을 통해서 factor로 전환하기
      detector.prop.factor <- cut(detector.prop.vector,
                                  breaks = seq(from = 0,
                                               to = 100,
                                               by = 100 / Nlevel),
                                  right = FALSE)
      
      # cut을 통해 변환된 factor를 적용하기
      for(k in 1:length(detector)){
        
        # 그 레벨에 해당하는 수들을 전부 그 레벨 값으로 바꾸기
        
        # 백분율이 100이라서, 해당 레벨이 NA값인 경우
        if(is.na(detector.prop.factor[k])){
          
          input[input == names(detector)[k]] <- '[all]'
          
          # 그 외
        }else{
          
          input[input == names(detector)[k]] <- as.character(detector.prop.factor[k])
          
        }
        
      }
      
    }
  
    # 다시 factor로 변환
    input <- as.factor(input)
    
    return(input)
    
}

# 레벨의 이름을 원하는 만큼 잘라주는 함수
CutLevels <- function(data.variable, start, end){
  
  data.variable <- data.variable %>% as.character()
  data.variable <- data.variable %>% str_sub(start = start, 
                                             end = end)
  data.variable <- data.variable %>% as.factor()
  
  return(data.variable)
  
}

# NA를 '미응답'(default)으로 변환하고, factor로 변환해주는 함수
NAtoFactor <- function(data.variable, NA.message = '미응답'){
  
  data.variable <- as.character(data.variable)
  
  data.variable[is.na(data.variable) == TRUE] <- NA.message
  
  data.variable <- as.factor(data.variable)
  
  return(data.variable)
  
}
```

데이터셋 불러오기
=================

``` {r}
dataset <- read.csv(file = 'trainset_mini.csv',
                    header = TRUE)

# 이름 제거하기
dataset <- dataset[,-1]

# HasDetections : 목표변수. factor로 변환
dataset$HasDetections <- as.factor(dataset$HasDetections)
dataset$HasDetections <- relevel(dataset$HasDetections, ref = '1')
```

1. DataSet<br><br><br>
----------------------

<div style = "color:red">
1.  데이터셋 출처
    </div>
    <https://www.kaggle.com/c/microsoft-malware-prediction>

<br>Kaggle Research Prediction Competition<br>( kaggle에서 주관하는 예측 대회)<br> 실제 데이터는 약 1500만행, 83개의 column을 가지고 있다.<br> 그 중에서 800만행은 목표변수를 포함하여 trainset으로 제공하고 있으며, 나머지 700만행에는 목표변수를 제외하여 testset으로 제공하고 있다.<br><br>목표변수는 HasDetections라는 컬럼으로서, 1과 0으로 이루어져 있기 때문에 우리들의 목적은 이진분류를 통한 악성코드 감염 여부 예측이라고 할 수 있겠다.<br><br> 우리조는 각 조원들의 컴퓨터 여건을 고려하여 trainset 데이터 중에서 **1%만을 sample함수로 추출하여 이것을 다시 0.7:0.3의 비율로 trainset과 validationset**으로 나누어서 예측에 대한 지표들을 확인하기로 하였다.<br><br><br><br>
<div style = "color:red">
1.  데이터셋 구조
    </div>
    **<데이터 변수 설명.hwp 참고>** 각 column의 설명 중 NA로서 마이크로스프트에서 특별한 주석을 제공하지 않은 변수 20개를 포함하여 제품이름(윈도우7~10)과 각종 Identifier들이 존재한다.<br>

**str을 통해 간단하게 확인해본 데이터셋**

``` {r}
str(dataset) # 데이터셋의 구조
```

<br><br> **1%로 샘플링한 데이터셋의 길이**

``` {r}
nrow(dataset) # 89155개
```

<br><br> **1%로 샘플링한 데이터셋의 목표변수의 1과 0의 비율**

``` {r}
dataset$HasDetections %>% table() %>% prop.table()
```

<br><br> **각 변수별 NA값 확인하기**

``` {r}
sapply(dataset, function(x) sum(is.na(x)))
```

<br><br> 전처리 하기에 앞서서, NA가 있어도 자동으로 처리해줄 수 있는 모형이 있고, NA가 없어야만 하는 모형도 있다.<br>여러가지 모형들의 성능을 평가하기 위해서는 NA는 어떻게든 처리해주는 편이 좋을 것 같다.<br> 실제로 어떤 IT기업이 소비자들의 악성코드 감염 여부를 예측할 때, 모든 데이터를 전부 조사하기는 힘들 것이다.<br> 컴퓨터에 일가견이 있는 사람들을 제외한 대다수에 사람들이 자신의 컴퓨터에 대해서 잘 알지 못하며, 옵션들을 함부로 건들이기를 꺼려한다. 또한, 실제로 사용자들을 대상으로 악성코드에 대한 예측을 하고자할 때, **기업이 미쳐 확인하지 못한 것들이 있을 것이다. 이것들을 모두 고려하여 최대한 정확한 예측을 하는 모형이 기업이 원하는 모형일 것이다.** 따라서 NA라고 삭제해버리는 것은 안 좋은 선택일수도 있다.<br><br> 우리조에서 생각해볼 NA 전처리문제 해결법<br> 1. 보류한다.<br> 2. 모두 제거한다.<br> **3. 제 3의 범주로 만든다. ( 범주형으로 만들어서 해결하기 )**<br> **4. 대체값을 찾는다. (단, int형으로 만들 수 있는 변수들만)**<br> 5. 기타 방법<br><br><br> 위에서 말했듯이, 1번의 NA를 단순히 보류하는 것은 데이터 낭비일 수도 있다. 그리고 2번의 NA를 모두 제거하는 것은 사실상 NA를 보류하는 것과 같은 말이다. 그리고 우리가 정말 고려해야할 방법은 3번과 4번일 것이다.<br> 5번의 경우는, 일단 모형을 만들어보고, 각 변수들 중 중요도가 높은 것들을 중심으로 전처리하는 방식 등이 있을 것이다. 이 방식들은 이 프로젝트가 끝난 후에 개인적으로 만들어볼 생각이다.<br><br><br><br>

2. 전처리 하기 전의 데이터셋 의사결정나무<br>
---------------------------------------------

전처리를 하지 않은 상태에서 만든 모형은 아마도 NA를 모두 제거한 상태와 같을 것이다. 위에서 해결법 1번과 2번에 해당하는 방식일 것이다.<br><br>

\*\* dataset을 trainset과 validationset으로 나누기\*\*

``` {r}
set.seed(123)

index <- sample(1:2,
                size = nrow(dataset),
                prob = c(0.7,0.3),
                replace = TRUE)

# t은 trainset, v는 validationset이다.
# 현재 testset의 목표변수를 알 수 없으므로, 어쩔 수 없이 dataset을 q1과 q2로 분리하여 예측률을 확인하도록 한다.
dataset.t <- dataset[index == 1, ]
dataset.v <- dataset[index == 2, ]
```

<br><br> \*\* 의사결정나무 모형 적합해보기\*\*

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

<br><br> **랜덤포레스트 모형 적합해보기**

``` {r}
# AvSigVersion
# AppVersion
# OsBuildLab
# Census_OSVersion

# 임시방편으로 레벨의 수가 많은 column을 제거하고 랜덤포레스트 모형에 적합하였다.

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

<br><br> 의사결정나무 보다도 상당히 낮은 민감도를 보이며, NA값들을 제거하고 보니, 남은 행들이 거의 없다시피 하였다. 따라서 전처리를 통해서, NA값을 범주로 만들거나, 특정 변수들을 int형으로 변환한 후, NA값을 대체값으로 대체하는 방법을 사용하기로 하였다.<br><br><br><br>

3. NA값이 있는 변수들을 범주형으로 전처리하기
---------------------------------------------

<br><br> 위에서 보았듯이, NA값을 전처리하지 않고, 그대로 사용한다면 좋은 예측 모형을 기대하기 어려울 것 같다. 그냥 50%확률로 찍어서 예측하는 것과 비슷한 정도이다.<br>**또한, 데이터를 확인하는 과정에서 NA값에 의미가 있는 경우를 확인하였다.** 예를 들자면, 변수 **IsProtected**의 경우에는 1일 때는 백신을 실행 중, 0일 때는 업데이트를 하지 않은 백신을 실행 중, **NA일 경우 백신을 사용하지 않는다.** 라는 의미가 된다.<br><br>따라서 NA를 범주로 처리해보기로 하였다.<br><br>

``` {r}
# 전처리할 데이터셋
dataset.q <- dataset
```

**버전을 담고 있는 factor**

``` {r}
# 버전을 담고 있는 factor
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

<br><br> **""라는 이름의 레벨을 가진 factor에 대해서 '미응답'으로 이름 바꾸기**

``` {r}
# "" 가 포함된 factor를 '미응답'으로 바꾸기
factor.name <- c('Census_PrimaryDiskTypeName',
                 'Census_ChassisTypeName',
                 'Census_PowerPlatformRoleName')

for(i in 1:3){
  
  dataset.q[,factor.name[i]] <- as.character(dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- ifelse(dataset.q[,factor.name[i]] == "", yes = "미응답", dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- as.factor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **NA값이 대다수인 변수**

``` {r}
# NA값이 대다수인 변수
factor.name <- c('DefaultBrowsersIdentifier',
                 'OrganizationIdentifier',
                 'Census_IsFlightingInternal',
                 'Census_ThresholdOptIn')

for(i in 1:length(factor.name)){
  
  dataset.q[,factor.name[i]] <- NAtoFactor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **기타 변수**

``` {r}
# 기타 변수
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

<br><br> **수치형, 혹은 범주형으로 전환할 수 있는 변수**<br> 이 변수들의 경우에는 2가지 모두 사용해서 확인해본다.

``` {r}
# int형으로 그대로 사용할 수 있고, 혹은 범주형으로 전환할 수 있는 변수
# 이 변수의 경우 2가지를 모두해서 확인해본다.

# 범주형으로 전환할 데이터셋
dataset.q.1 <- dataset.q
# int형 그대로 사용할 데이터셋
dataset.q.2 <- dataset.q


factor.name <- c('Census_ProcessorCoreCount',
                 'Census_PrimaryDiskTotalCapacity',
                 'Census_SystemVolumeTotalCapacity',
                 'Census_TotalPhysicalRAM',
                 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                 'Census_InternalPrimaryDisplayResolutionHorizontal',
                 'Census_InternalPrimaryDisplayResolutionVertical')


# (1) NA처리하고, 범주형으로 변환하기
for(i in 1:length(factor.name)){
  
  dataset.q.1[,factor.name[i]] <- NAtoFactor(dataset.q.1[,factor.name[i]])
  
}

# (2) NA를 대체값으로 대체하기
for(i in 1:length(factor.name)){

  dataset.q.2[is.na(dataset.q.2[,factor.name[i]]),
              factor.name[i]] <- summary(dataset.q.2[,factor.name[i]])[3]

}
```

<br><br> **전처리된 데이터셋 1번의 NA와 레벨수 확인하기**

``` {r}
# NA값과 레벨수 확인하기
factor.name <- colnames(dataset.q.1)

for(i in 1:length(factor.name)){
  
  cat('변수명 : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('레벨수 : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

<br><br> **전처리된 데이터셋 2번의 NA와 레벨수 확인하기**

``` {r}
# NA값과 레벨수 확인하기
factor.name <- colnames(dataset.q.2)

for(i in 1:length(factor.name)){
  
  cat('변수명 : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('레벨수 : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

4.전처리한 데이터셋으로 의사결정나무 적합하기<br><br>
-----------------------------------------------------

**전처리한 데이터셋으로 의사결정나무 적합하기 1**

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

<br><br> **전처리한 데이터셋으로 의사결정나무 적합하기 2**

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

5. 변수 레벨 축소해보기
-----------------------

일부 변수들의 NA값을 Mean으로 대체할 때의 정확도가 조금 더 높았다. 하지만, 아무런 전처리를 하지 않은 상태에서 적합한 의사결정나무 모형보다는 부족한 수치를 보이고 있다. identifier와 같은 변수들을 factor로 변환하면서 수많은 레벨이 생성되어 오히려 전체적인 지표들의 수치가 감소한 것 같다.<br><br>따라서 레벨의 수를 어느정도 줄인다면, 더 좋은 결과가 나올 것이라고 판단했다.<br><br>HasDetections(목표변수)의 1과 0의 빈도가 유사한 것끼리 그룹으로 묶어서 레벨의 수가 50개 이상인 컬럼을 50개 이하의 레벨로 줄여보도록 하였다.<br><br> **50개 이상의 레벨의 수 줄이기**

``` {r}
factor.name <- colnames(dataset.q.1)
Com.factor.name <- c() # 레벨이 50개 이상인 컬럼명


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.1[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.1)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.1[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.1$HasDetections, input = dataset.q.1[,Com.factor.name[i]])
  
}

# 레벨 수 다시 한 번 확인해보기
for(i in 1:length(factor.name)){
  
  cat('변수명 : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('레벨수 : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

``` {r}
factor.name <- colnames(dataset.q.2)
Com.factor.name <- c() # 레벨이 50개 이상인 컬럼명


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.2[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.2)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.2[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.2$HasDetections, input = dataset.q.2[,Com.factor.name[i]])
  
}

# 레벨 수 다시 한 번 확인해보기
for(i in 1:length(factor.name)){
  
  cat('변수명 : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('레벨수 : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

6. 축소한 데이터셋으로 의사결정나무, 랜덤포레스트 적합하기
----------------------------------------------------------

<br><br> 이제부터 의사결정나무를 적합해보고, 랜덤포레스트는 튜닝까지 해보겠다.<br><br><br><br> **의사결정나무 적합하기 1**

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

# 혼동행렬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc 확인용
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# 비용복잡도 표 출력
printcp(fitTree)
```

**의사결정나무 적합하기 2**

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

# 혼동행렬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc 확인용
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# 비용복잡도 표 출력
printcp(fitTree)
```

<br><br><br> 레벨의 수를 축소한 이후로는 모든 컬럼을 범주형으로 변경한 데이터셋에서 더 높은 F1 점수와 auroc 값이 나왔다. 상황이 역전되었다. 각각 모형에서 가장 마지막 분리에서 xerror가 가장 낮게 나왔으므로, 가지치기를 할 필요가 없었다. 그렇다면 이제 모든 컬럼을 범주형으로 변경한 데이터셋으로 랜덤포레스트 모형을 적합해보기로 하였다.<br><br> **랜덤포레스트 모형 간단하게 적합해보기**

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

# 모형 적합 결과 확인하기 ( 오분류율 확인하기 )
print(fitRFC)

# 변수 중요도 출력하기
importance(fitRFC)

# 마진 그래프
plot(margin(fitRFC))

# 혼동행렬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc 확인용
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> MeanDecreaseAccuracy에서 크게 영향을 주는 변수는,<br> AppVersion 13.2078028<br> AVProductStatesIdentifier 21.1545553<br> CountryIdentifier 10.5903571<br> CityIdentifier 51.7872658<br> SmartScreen 31.7732623<br> Census\_OEMModelIdentifier 40.9204674<br> Census\_ProcessorModelIdentifier 13.0899241<br> Census\_SystemVolumeTotalCapacity 103.9377516<br> Census\_InternalBatteryNumberOfCharges 16.4877933<br> Census\_OSInstallLanguageIdentifier 13.7160384<br> Census\_FirmwareVersionIdentifier 31.2409193<br> Wdft\_RegionIdentifier 12.5699366<br><br> 이것들이다. 각 변수들의 의미를 간단하게 알아보자면,<br> 1. windows defender의 버전<br> 2. 안티 바이러스 백신의 버전<br> 3. 국가코드, 도시코드, 지역코드, 언어코드<br> 4. smartscreen(윈도우 10의 방화벽 기능)<br> 5. OEM의 종류(운영체제를 대량으로 구매하여 설치하는 방식)<br> 6. CPU의 모델 명<br> 7. 운영체제가 깔려있는 파티션의 크기<br> 8. 방화벽의 버전<br> 이다. 그리고 혼동행렬을 확인해본 결과, 이 모형은 감염되지 않은 PC를 더 잘 찾아내는 특이도와 정밀도가 높은 모형이다.<br> 이 모형은 정밀도가 상당히 높은 편이지만, 악성 코드에 감염된 컴퓨터를 감염되지 않았다고 판단하는 경우가 상당하기 때문에 아쉬운 모형이다.<br> 그렇기 때문에 튜닝을 한다면, 민감도가 더 높은 모형이 나올 수 있지 않을까 생각을 해보았다.<br> **랜덤포레스트 모형 튜닝해보기**

``` {r}
# # 총 32개의 조합
# # 와 이런 좋은 함수가 있었어?
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
#   # 예측값
#   trPred <- fit$test$predicted
#   # 실제값
#   trReal <- dataset.q.1.v$HasDetections
#   # 혼동행렬
#   con <- confusionMatrix(trPred, trReal, positive = '1')
#   
#   # 오분류수
#   mcSum <- sum(fit$predicted != dataset.q.1.t$HasDetections)
#   # 오분류율
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

<br><br> 시간이 없는 관계로 매우 간단하게 그리드 서치를 하였다.<br> ntree = 500, mtry = 13정도로 랜덤포레스트 모형을 적합하면 적당할 것 같다.<br><br> **튜닝한 랜덤포레스트 모형**

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

# 모형 적합 결과 확인하기 ( 오분류율 확인하기 )
print(fitRFC)

# 변수 중요도 출력하기
importance(fitRFC)

# 마진 그래프
plot(margin(fitRFC))

# 혼동행렬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc 확인용
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

7. 번외 - XGBoost 사용해보기<br><br>
------------------------------------

각종 대회에서 높은 점수를 기록하는 머신러닝 알고리즘 중 하나인, XGBoost를 사용하여 성능을 평가해보기로 하였다.<br> **라벨링, 더미변수 함수**

``` {r}
# 라벨링 함수 ( numeric vector로 반환 )
MakingLabel <- function(data.variable){
  
  for(i in 1:nlevels(data.variable)){
  
    levels(data.variable)[i] <- i
  
  }
  
  data.variable <- as.numeric(data.variable)
  return(data.variable)
}

# 더미변수 만드는 함수 ( 더미변수들을 data.frame으로 반환 )
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

# 사용예시
example.label <- MakingLabel(dataset.q.1$ProductName)
example.dummy <- MakingDummy(dataset.q.1$ProductName,
                             'result')
```

<br><br> **데이터셋 변환하기**

``` {r}
# 데이터셋
dataset.n <- dataset.q.1

HasDetections <- dataset.n$HasDetections %>% as.numeric()
HasDetections <- ifelse(HasDetections == 2,0,1)

dataset.n.label <- dataset.n[, -82]
dataset.n.dummy <- dataset.n

# dataset.n.dummy의 레벨 수 자르기
factor.name <- colnames(dataset.n.dummy)
Com.factor.name <- c() # 레벨이 50개 이상인 컬럼명


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

# 레벨 수 다시 확인하기
for(i in 1:length(factor.name)){
  
  cat('변수명 : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.n.dummy[,factor.name[i]]), '\n')
  cat('레벨수 : ', nlevels(dataset.n.dummy[,factor.name[i]]), 
      '\n\n')
  
}

# 목표변수만 제거하기
dataset.n.dummy <- dataset.n.dummy[, -82]

# factor.name에서 목표변수 제거하기
factor.name <- colnames(dataset.n)[-82]

# 1. 라벨링하기
for(i in 1:length(factor.name)){
  
  dataset.n.label[, factor.name[i]] <- MakingLabel(dataset.n.label[, factor.name[i]])
  
}

dataset.n.label <- cbind(dataset.n.label, HasDetections)
ncol(dataset.n.label)

# 2. 더미변수화
for(i in 1:length(factor.name)){
  
  dataset.n.dummy <- cbind(dataset.n.dummy,
                           MakingDummy(dataset.n.dummy[,factor.name[i]],
                                       factor.name[i]))
  
}

dataset.n.dummy <- dataset.n.dummy[,-c(1:81)]
dataset.n.dummy <- cbind(dataset.n.dummy, HasDetections)
ncol(dataset.n.dummy)

# trainset과 validationset으로 나누기
dataset.n.label.t <- dataset.n.label[index == 1, ]
dataset.n.dummy.t <- dataset.n.dummy[index == 1, ]

dataset.n.label.v <- dataset.n.label[index == 2, ]
dataset.n.dummy.v <- dataset.n.dummy[index == 2, ]
```

<br><br> **XGBoost 사용해보기 1**

``` {r}
# 1. 라벨링한 데이터셋
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.label.t[ ,-82]),
                      label= as.matrix(dataset.n.label.t[ , 82]))


# 파라미터
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

# xgboost 모형 적합하기
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.label.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.label.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# 혼동행렬
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc 확인용
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> **XGBoost 사용해보기 2**

``` {r}
# 2. 더미변수로 만든 데이터셋
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.dummy.t[ ,-82]),
                      label= as.matrix(dataset.n.dummy.t[ , 82]))


# 파라미터
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

# xgboost 모형 적합하기
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.dummy.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.dummy.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# 혼동행렬
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc 확인용
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

#### ?<br><br>

8. 마무리
---------

우리들의 관심분야인 악성코드에 대한 예측 데이터셋에 대해서는 어느정도 개요가 잡힌 상태였지만, 튜닝과 전처리에 있어서 여러가지 아쉬운 한계점들이 있었다.<br> 1. 1500만행의 실제 데이터에서는 testset에는 있지만, trainset에는 없는 Identifier가 있기 때문에, 약 5~10%정도의 정보를 사용하지 못하는 것을 확인했다.<br><br> 2. 실제 데이터에서는 현재의 방법 중 몇군대를 수정해야하고, 그로 인해서 치명적인 오차가 발생할 수도 있다.<br><br> 3. 컴퓨터의 사양 문제로 실제 데이터는 전처리 시간에만 7일 이상이 소요될 것이다.<br><br> 4. 그렇기 때문에, 실제 데이터를 다뤄보지 못하였고, 다양한 방법들을 고민하기에는 한계가 있었다. 이 과정이 끝난 후에 개인적으로 이 대회를 마무리 지을 생각이다.
