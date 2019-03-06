ë¨¸ì‹ ?Ÿ¬?‹ ì¡°ë³„ ?”„ë¡œì ?Š¸
================
?˜¤ì¤€?Š¹-?œ¤?œ˜?˜-ê¹€?ˆ˜?˜„-ê°•ë?¼ê¸°
2019?…„ 2?›” 26?¼

`{r setup, include=FALSE} knitr::opts_chunk$set(echo = TRUE)`

?•„?š” ?Œ¨?‚¤ì§€ ë¶ˆëŸ¬?˜¤ê¸? & ?ž‘?—…ê³µê°„ ?„¤? •?•˜ê¸?
----------------------------------------

``` {r}
library(tidyverse)
library(dplyr)
library(randomForest) # ?žœ?¤?¬? ˆ?Š¤?Š¸
library(rpart) # ?˜?‚¬ê²°ì •?‚˜ë¬? 
library(caret) # ?˜¼?™?–‰? ¬?— ?•„?š”?•œ ?Œ¨?‚¤ì§€
library(MLmetrics) # F1 ? ?ˆ˜?— ?•„?š”?•œ ?Œ¨?‚¤ì§€
library(purrr)
library(e1071)
library(xgboost) # xgboost
library(ROCR) # roc ì»¤ë¸Œ?— ?•„?š”?•œ ?Œ¨?‚¤ì§€
library(pROC) # auroc

setwd('d:/fastcampus/') # ?ž‘?—…ê³µê°„ ?„¤? •?•˜ê¸?
getwd() # ?„¤? •?œ ?ž‘?—…ê³µê°„ ?™•?¸?•˜ê¸?
```

``` {r}
# ? ˆë²¨ì„ ë¹ˆë„?— ?”°?¼ 50ê°œë¡œ ?••ì¶•í•˜?Š” ?•¨?ˆ˜
# ?¸?ž : ëª©í‘œë³€?ˆ˜(trainset.q$HasDetections), ?ž…? ¥ë³€?ˆ˜(trainset.q[,factor.name[i]])
CompressLevels <- function(object, input, Nlevel = 50){
    
    # factor?˜ ? ˆë²¨ì— ?”°ë¥? ëª©í‘œë³€?ˆ˜?˜ 0ê³? 1?˜ ë¹ˆë„ë¥? ì§€?—­ë³€?ˆ˜?— ?• ?‹¹
    detector <- by(object,
                   input,
                   table)
    
    # ? ˆë²? ì¶•ì†Œë¥? ?œ„?•´?„œ ?ž„?‹œ? ?œ¼ë¡? characterë¡? ë³€?™˜
    input <- as.character(input)
    
    # ? ˆë²¨ì˜ ?ˆ˜ê°€ 50ê°? ?´?ƒ?¸ ê²½ìš°
    if(length(detector) > Nlevel){
      
      # ? ˆë²¨ì— ?”°ë¥? ë°±ë¶„?œ¨?„ ?‹´?„ ?ƒˆë¡œìš´ ê°ì²´
      detector.prop.vector <- c()
      
      # ê°? ? ˆë²¨ì— ?”°ë¥? ë¹ˆë„ë¥? ë°±ë¶„?œ¨ë¡? ? „?™˜?•˜ê¸?
      for (k in 1:length(detector)) {
        
        detector.prop <- 100 * detector[[k]][1] / (detector[[k]][1] + detector[[k]][2])
        
        detector.prop.vector <- rbind(detector.prop.vector, detector.prop)
        
      }
      
      # cut?„ ?†µ?•´?„œ factorë¡? ? „?™˜?•˜ê¸?
      detector.prop.factor <- cut(detector.prop.vector,
                                  breaks = seq(from = 0,
                                               to = 100,
                                               by = 100 / Nlevel),
                                  right = FALSE)
      
      # cut?„ ?†µ?•´ ë³€?™˜?œ factorë¥? ? ?š©?•˜ê¸?
      for(k in 1:length(detector)){
        
        # ê·? ? ˆë²¨ì— ?•´?‹¹?•˜?Š” ?ˆ˜?“¤?„ ? „ë¶€ ê·? ? ˆë²? ê°’ìœ¼ë¡? ë°”ê¾¸ê¸?
        
        # ë°±ë¶„?œ¨?´ 100?´?¼?„œ, ?•´?‹¹ ? ˆë²¨ì´ NAê°’ì¸ ê²½ìš°
        if(is.na(detector.prop.factor[k])){
          
          input[input == names(detector)[k]] <- '[all]'
          
          # ê·? ?™¸
        }else{
          
          input[input == names(detector)[k]] <- as.character(detector.prop.factor[k])
          
        }
        
      }
      
    }
  
    # ?‹¤?‹œ factorë¡? ë³€?™˜
    input <- as.factor(input)
    
    return(input)
    
}

# ? ˆë²¨ì˜ ?´ë¦„ì„ ?›?•˜?Š” ë§Œí¼ ?ž˜?¼ì£¼ëŠ” ?•¨?ˆ˜
CutLevels <- function(data.variable, start, end){
  
  data.variable <- data.variable %>% as.character()
  data.variable <- data.variable %>% str_sub(start = start, 
                                             end = end)
  data.variable <- data.variable %>% as.factor()
  
  return(data.variable)
  
}

# NAë¥? 'ë¯¸ì‘?‹µ'(default)?œ¼ë¡? ë³€?™˜?•˜ê³?, factorë¡? ë³€?™˜?•´ì£¼ëŠ” ?•¨?ˆ˜
NAtoFactor <- function(data.variable, NA.message = 'ë¯¸ì‘?‹µ'){
  
  data.variable <- as.character(data.variable)
  
  data.variable[is.na(data.variable) == TRUE] <- NA.message
  
  data.variable <- as.factor(data.variable)
  
  return(data.variable)
  
}
```

?°?´?„°?…‹ ë¶ˆëŸ¬?˜¤ê¸?
=================

``` {r}
dataset <- read.csv(file = 'trainset_mini.csv',
                    header = TRUE)

# ?´ë¦? ? œê±°í•˜ê¸?
dataset <- dataset[,-1]

# HasDetections : ëª©í‘œë³€?ˆ˜. factorë¡? ë³€?™˜
dataset$HasDetections <- as.factor(dataset$HasDetections)
dataset$HasDetections <- relevel(dataset$HasDetections, ref = '1')
```

1. DataSet<br><br><br>
----------------------

<div style = "color:red">
1.  ?°?´?„°?…‹ ì¶œì²˜
    </div>
    <https://www.kaggle.com/c/microsoft-malware-prediction>

<br>Kaggle Research Prediction Competition<br>( kaggle?—?„œ ì£¼ê?€?•˜?Š” ?˜ˆì¸? ??€?šŒ)<br> ?‹¤? œ ?°?´?„°?Š” ?•½ 1500ë§Œí–‰, 83ê°œì˜ column?„ ê°€ì§€ê³? ?žˆ?‹¤.<br> ê·? ì¤‘ì—?„œ 800ë§Œí–‰??€ ëª©í‘œë³€?ˆ˜ë¥? ?¬?•¨?•˜?—¬ trainset?œ¼ë¡? ? œê³µí•˜ê³? ?žˆ?œ¼ë©?, ?‚˜ë¨¸ì?€ 700ë§Œí–‰?—?Š” ëª©í‘œë³€?ˆ˜ë¥? ? œ?™¸?•˜?—¬ testset?œ¼ë¡? ? œê³µí•˜ê³? ?žˆ?‹¤.<br><br>ëª©í‘œë³€?ˆ˜?Š” HasDetections?¼?Š” ì»¬ëŸ¼?œ¼ë¡œì„œ, 1ê³? 0?œ¼ë¡? ?´ë£¨ì–´? ¸ ?žˆê¸? ?•Œë¬¸ì— ?š°ë¦¬ë“¤?˜ ëª©ì ??€ ?´ì§„ë¶„ë¥˜ë?? ?†µ?•œ ?•…?„±ì½”ë“œ ê°ì—¼ ?—¬ë¶€ ?˜ˆì¸¡ì´?¼ê³? ?•  ?ˆ˜ ?žˆê² ë‹¤.<br><br> ?š°ë¦¬ì¡°?Š” ê°? ì¡°ì›?“¤?˜ ì»´í“¨?„° ?—¬ê±´ì„ ê³ ë ¤?•˜?—¬ trainset ?°?´?„° ì¤‘ì—?„œ **1%ë§Œì„ sample?•¨?ˆ˜ë¡? ì¶”ì¶œ?•˜?—¬ ?´ê²ƒì„ ?‹¤?‹œ 0.7:0.3?˜ ë¹„ìœ¨ë¡? trainsetê³? validationset**?œ¼ë¡? ?‚˜?ˆ„?–´?„œ ?˜ˆì¸¡ì— ??€?•œ ì§€?‘œ?“¤?„ ?™•?¸?•˜ê¸°ë¡œ ?•˜??€?‹¤.<br><br><br><br>
<div style = "color:red">
1.  ?°?´?„°?…‹ êµ¬ì¡°
    </div>
    **<?°?´?„° ë³€?ˆ˜ ?„¤ëª?.hwp ì°¸ê³ >** ê°? column?˜ ?„¤ëª? ì¤? NAë¡œì„œ ë§ˆì´?¬ë¡œìŠ¤?”„?Š¸?—?„œ ?Š¹ë³„í•œ ì£¼ì„?„ ? œê³µí•˜ì§€ ?•Š??€ ë³€?ˆ˜ 20ê°œë?? ?¬?•¨?•˜?—¬ ? œ?’ˆ?´ë¦?(?œˆ?„?š°7~10)ê³? ê°ì¢… Identifier?“¤?´ ì¡´ìž¬?•œ?‹¤.<br>

**str?„ ?†µ?•´ ê°„ë‹¨?•˜ê²? ?™•?¸?•´ë³? ?°?´?„°?…‹**

``` {r}
str(dataset) # ?°?´?„°?…‹?˜ êµ¬ì¡°
```

<br><br> **1%ë¡? ?ƒ˜?”Œë§í•œ ?°?´?„°?…‹?˜ ê¸¸ì´**

``` {r}
nrow(dataset) # 89155ê°?
```

<br><br> **1%ë¡? ?ƒ˜?”Œë§í•œ ?°?´?„°?…‹?˜ ëª©í‘œë³€?ˆ˜?˜ 1ê³? 0?˜ ë¹„ìœ¨**

``` {r}
dataset$HasDetections %>% table() %>% prop.table()
```

<br><br> **ê°? ë³€?ˆ˜ë³? NAê°? ?™•?¸?•˜ê¸?**

``` {r}
sapply(dataset, function(x) sum(is.na(x)))
```

<br><br> ? „ì²˜ë¦¬ ?•˜ê¸°ì— ?•ž?„œ?„œ, NAê°€ ?žˆ?–´?„ ?ž?™?œ¼ë¡? ì²˜ë¦¬?•´ì¤? ?ˆ˜ ?žˆ?Š” ëª¨í˜•?´ ?žˆê³?, NAê°€ ?—†?–´?•¼ë§? ?•˜?Š” ëª¨í˜•?„ ?žˆ?‹¤.<br>?—¬?Ÿ¬ê°€ì§€ ëª¨í˜•?“¤?˜ ?„±?Š¥?„ ?‰ê°€?•˜ê¸? ?œ„?•´?„œ?Š” NA?Š” ?–´?–»ê²Œë“  ì²˜ë¦¬?•´ì£¼ëŠ” ?Ž¸?´ ì¢‹ì„ ê²? ê°™ë‹¤.<br> ?‹¤? œë¡? ?–´?–¤ ITê¸°ì—…?´ ?†Œë¹„ìž?“¤?˜ ?•…?„±ì½”ë“œ ê°ì—¼ ?—¬ë¶€ë¥? ?˜ˆì¸¡í•  ?•Œ, ëª¨ë“  ?°?´?„°ë¥? ? „ë¶€ ì¡°ì‚¬?•˜ê¸°ëŠ” ?ž˜?“¤ ê²ƒì´?‹¤.<br> ì»´í“¨?„°?— ?¼ê°€ê²¬ì´ ?žˆ?Š” ?‚¬?žŒ?“¤?„ ? œ?™¸?•œ ??€?‹¤?ˆ˜?— ?‚¬?žŒ?“¤?´ ?ž?‹ ?˜ ì»´í“¨?„°?— ??€?•´?„œ ?ž˜ ?•Œì§€ ëª»í•˜ë©?, ?˜µ?…˜?“¤?„ ?•¨ë¶€ë¡? ê±´ë“¤?´ê¸°ë?? êº¼ë ¤?•œ?‹¤. ?˜?•œ, ?‹¤? œë¡? ?‚¬?š©?ž?“¤?„ ??€?ƒ?œ¼ë¡? ?•…?„±ì½”ë“œ?— ??€?•œ ?˜ˆì¸¡ì„ ?•˜ê³ ìž?•  ?•Œ, **ê¸°ì—…?´ ë¯¸ì³ ?™•?¸?•˜ì§€ ëª»í•œ ê²ƒë“¤?´ ?žˆ?„ ê²ƒì´?‹¤. ?´ê²ƒë“¤?„ ëª¨ë‘ ê³ ë ¤?•˜?—¬ ìµœë?€?•œ ? •?™•?•œ ?˜ˆì¸¡ì„ ?•˜?Š” ëª¨í˜•?´ ê¸°ì—…?´ ?›?•˜?Š” ëª¨í˜•?¼ ê²ƒì´?‹¤.** ?”°?¼?„œ NA?¼ê³? ?‚­? œ?•´ë²„ë¦¬?Š” ê²ƒì?€ ?•ˆ ì¢‹ì?€ ?„ ?ƒ?¼?ˆ˜?„ ?žˆ?‹¤.<br><br> ?š°ë¦¬ì¡°?—?„œ ?ƒê°í•´ë³? NA ? „ì²˜ë¦¬ë¬¸ì œ ?•´ê²°ë²•<br> 1. ë³´ë¥˜?•œ?‹¤.<br> 2. ëª¨ë‘ ? œê±°í•œ?‹¤.<br> **3. ? œ 3?˜ ë²”ì£¼ë¡? ë§Œë“ ?‹¤. ( ë²”ì£¼?˜•?œ¼ë¡? ë§Œë“¤?–´?„œ ?•´ê²°í•˜ê¸? )**<br> **4. ??€ì²´ê°’?„ ì°¾ëŠ”?‹¤. (?‹¨, int?˜•?œ¼ë¡? ë§Œë“¤ ?ˆ˜ ?žˆ?Š” ë³€?ˆ˜?“¤ë§?)**<br> 5. ê¸°í?€ ë°©ë²•<br><br><br> ?œ„?—?„œ ë§í–ˆ?“¯?´, 1ë²ˆì˜ NAë¥? ?‹¨?ˆœ?žˆ ë³´ë¥˜?•˜?Š” ê²ƒì?€ ?°?´?„° ?‚­ë¹„ì¼ ?ˆ˜?„ ?žˆ?‹¤. ê·¸ë¦¬ê³? 2ë²ˆì˜ NAë¥? ëª¨ë‘ ? œê±°í•˜?Š” ê²ƒì?€ ?‚¬?‹¤?ƒ NAë¥? ë³´ë¥˜?•˜?Š” ê²ƒê³¼ ê°™ì?€ ë§ì´?‹¤. ê·¸ë¦¬ê³? ?š°ë¦¬ê?€ ? •ë§? ê³ ë ¤?•´?•¼?•  ë°©ë²•??€ 3ë²ˆê³¼ 4ë²ˆì¼ ê²ƒì´?‹¤.<br> 5ë²ˆì˜ ê²½ìš°?Š”, ?¼?‹¨ ëª¨í˜•?„ ë§Œë“¤?–´ë³´ê³ , ê°? ë³€?ˆ˜?“¤ ì¤? ì¤‘ìš”?„ê°€ ?†’??€ ê²ƒë“¤?„ ì¤‘ì‹¬?œ¼ë¡? ? „ì²˜ë¦¬?•˜?Š” ë°©ì‹ ?“±?´ ?žˆ?„ ê²ƒì´?‹¤. ?´ ë°©ì‹?“¤??€ ?´ ?”„ë¡œì ?Š¸ê°€ ??‚œ ?›„?— ê°œì¸? ?œ¼ë¡? ë§Œë“¤?–´ë³? ?ƒê°ì´?‹¤.<br><br><br><br>

2. ? „ì²˜ë¦¬ ?•˜ê¸? ? „?˜ ?°?´?„°?…‹ ?˜?‚¬ê²°ì •?‚˜ë¬?<br>
---------------------------------------------

? „ì²˜ë¦¬ë¥? ?•˜ì§€ ?•Š??€ ?ƒ?ƒœ?—?„œ ë§Œë“  ëª¨í˜•??€ ?•„ë§ˆë„ NAë¥? ëª¨ë‘ ? œê±°í•œ ?ƒ?ƒœ??€ ê°™ì„ ê²ƒì´?‹¤. ?œ„?—?„œ ?•´ê²°ë²• 1ë²ˆê³¼ 2ë²ˆì— ?•´?‹¹?•˜?Š” ë°©ì‹?¼ ê²ƒì´?‹¤.<br><br>

\*\* dataset?„ trainsetê³? validationset?œ¼ë¡? ?‚˜?ˆ„ê¸?\*\*

``` {r}
set.seed(123)

index <- sample(1:2,
                size = nrow(dataset),
                prob = c(0.7,0.3),
                replace = TRUE)

# t??€ trainset, v?Š” validationset?´?‹¤.
# ?˜„?ž¬ testset?˜ ëª©í‘œë³€?ˆ˜ë¥? ?•Œ ?ˆ˜ ?—†?œ¼ë¯€ë¡?, ?–´ì©? ?ˆ˜ ?—†?´ dataset?„ q1ê³? q2ë¡? ë¶„ë¦¬?•˜?—¬ ?˜ˆì¸¡ë¥ ?„ ?™•?¸?•˜?„ë¡? ?•œ?‹¤.
dataset.t <- dataset[index == 1, ]
dataset.v <- dataset[index == 2, ]
```

<br><br> \*\* ?˜?‚¬ê²°ì •?‚˜ë¬? ëª¨í˜• ? ?•©?•´ë³´ê¸°\*\*

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

<br><br> **?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜• ? ?•©?•´ë³´ê¸°**

``` {r}
# AvSigVersion
# AppVersion
# OsBuildLab
# Census_OSVersion

# ?ž„?‹œë°©íŽ¸?œ¼ë¡? ? ˆë²¨ì˜ ?ˆ˜ê°€ ë§Žì?€ column?„ ? œê±°í•˜ê³? ?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜•?— ? ?•©?•˜??€?‹¤.

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

<br><br> ?˜?‚¬ê²°ì •?‚˜ë¬? ë³´ë‹¤?„ ?ƒ?‹¹?žˆ ?‚®??€ ë¯¼ê°?„ë¥? ë³´ì´ë©?, NAê°’ë“¤?„ ? œê±°í•˜ê³? ë³´ë‹ˆ, ?‚¨??€ ?–‰?“¤?´ ê±°ì˜ ?—†?‹¤?‹œ?”¼ ?•˜??€?‹¤. ?”°?¼?„œ ? „ì²˜ë¦¬ë¥? ?†µ?•´?„œ, NAê°’ì„ ë²”ì£¼ë¡? ë§Œë“¤ê±°ë‚˜, ?Š¹? • ë³€?ˆ˜?“¤?„ int?˜•?œ¼ë¡? ë³€?™˜?•œ ?›„, NAê°’ì„ ??€ì²´ê°’?œ¼ë¡? ??€ì²´í•˜?Š” ë°©ë²•?„ ?‚¬?š©?•˜ê¸°ë¡œ ?•˜??€?‹¤.<br><br><br><br>

3. NAê°’ì´ ?žˆ?Š” ë³€?ˆ˜?“¤?„ ë²”ì£¼?˜•?œ¼ë¡? ? „ì²˜ë¦¬?•˜ê¸?
---------------------------------------------

<br><br> ?œ„?—?„œ ë³´ì•˜?“¯?´, NAê°’ì„ ? „ì²˜ë¦¬?•˜ì§€ ?•Šê³?, ê·¸ë?€ë¡? ?‚¬?š©?•œ?‹¤ë©? ì¢‹ì?€ ?˜ˆì¸? ëª¨í˜•?„ ê¸°ë?€?•˜ê¸? ?–´? ¤?š¸ ê²? ê°™ë‹¤. ê·¸ëƒ¥ 50%?™•ë¥ ë¡œ ì°ì–´?„œ ?˜ˆì¸¡í•˜?Š” ê²ƒê³¼ ë¹„ìŠ·?•œ ? •?„?´?‹¤.<br>**?˜?•œ, ?°?´?„°ë¥? ?™•?¸?•˜?Š” ê³¼ì •?—?„œ NAê°’ì— ?˜ë¯¸ê?€ ?žˆ?Š” ê²½ìš°ë¥? ?™•?¸?•˜??€?‹¤.** ?˜ˆë¥? ?“¤?žë©?, ë³€?ˆ˜ **IsProtected**?˜ ê²½ìš°?—?Š” 1?¼ ?•Œ?Š” ë°±ì‹ ?„ ?‹¤?–‰ ì¤?, 0?¼ ?•Œ?Š” ?—…?°?´?Š¸ë¥? ?•˜ì§€ ?•Š??€ ë°±ì‹ ?„ ?‹¤?–‰ ì¤?, **NA?¼ ê²½ìš° ë°±ì‹ ?„ ?‚¬?š©?•˜ì§€ ?•Š?Š”?‹¤.** ?¼?Š” ?˜ë¯¸ê?€ ?œ?‹¤.<br><br>?”°?¼?„œ NAë¥? ë²”ì£¼ë¡? ì²˜ë¦¬?•´ë³´ê¸°ë¡? ?•˜??€?‹¤.<br><br>

``` {r}
# ? „ì²˜ë¦¬?•  ?°?´?„°?…‹
dataset.q <- dataset
```

**ë²„ì „?„ ?‹´ê³? ?žˆ?Š” factor**

``` {r}
# ë²„ì „?„ ?‹´ê³? ?žˆ?Š” factor
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

<br><br> **""?¼?Š” ?´ë¦„ì˜ ? ˆë²¨ì„ ê°€ì§? factor?— ??€?•´?„œ 'ë¯¸ì‘?‹µ'?œ¼ë¡? ?´ë¦? ë°”ê¾¸ê¸?**

``` {r}
# "" ê°€ ?¬?•¨?œ factorë¥? 'ë¯¸ì‘?‹µ'?œ¼ë¡? ë°”ê¾¸ê¸?
factor.name <- c('Census_PrimaryDiskTypeName',
                 'Census_ChassisTypeName',
                 'Census_PowerPlatformRoleName')

for(i in 1:3){
  
  dataset.q[,factor.name[i]] <- as.character(dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- ifelse(dataset.q[,factor.name[i]] == "", yes = "ë¯¸ì‘?‹µ", dataset.q[,factor.name[i]])
  dataset.q[,factor.name[i]] <- as.factor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **NAê°’ì´ ??€?‹¤?ˆ˜?¸ ë³€?ˆ˜**

``` {r}
# NAê°’ì´ ??€?‹¤?ˆ˜?¸ ë³€?ˆ˜
factor.name <- c('DefaultBrowsersIdentifier',
                 'OrganizationIdentifier',
                 'Census_IsFlightingInternal',
                 'Census_ThresholdOptIn')

for(i in 1:length(factor.name)){
  
  dataset.q[,factor.name[i]] <- NAtoFactor(dataset.q[,factor.name[i]])
  
}
```

<br><br> **ê¸°í?€ ë³€?ˆ˜**

``` {r}
# ê¸°í?€ ë³€?ˆ˜
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

<br><br> **?ˆ˜ì¹˜í˜•, ?˜¹??€ ë²”ì£¼?˜•?œ¼ë¡? ? „?™˜?•  ?ˆ˜ ?žˆ?Š” ë³€?ˆ˜**<br> ?´ ë³€?ˆ˜?“¤?˜ ê²½ìš°?—?Š” 2ê°€ì§€ ëª¨ë‘ ?‚¬?š©?•´?„œ ?™•?¸?•´ë³¸ë‹¤.

``` {r}
# int?˜•?œ¼ë¡? ê·¸ë?€ë¡? ?‚¬?š©?•  ?ˆ˜ ?žˆê³?, ?˜¹??€ ë²”ì£¼?˜•?œ¼ë¡? ? „?™˜?•  ?ˆ˜ ?žˆ?Š” ë³€?ˆ˜
# ?´ ë³€?ˆ˜?˜ ê²½ìš° 2ê°€ì§€ë¥? ëª¨ë‘?•´?„œ ?™•?¸?•´ë³¸ë‹¤.

# ë²”ì£¼?˜•?œ¼ë¡? ? „?™˜?•  ?°?´?„°?…‹
dataset.q.1 <- dataset.q
# int?˜• ê·¸ë?€ë¡? ?‚¬?š©?•  ?°?´?„°?…‹
dataset.q.2 <- dataset.q


factor.name <- c('Census_ProcessorCoreCount',
                 'Census_PrimaryDiskTotalCapacity',
                 'Census_SystemVolumeTotalCapacity',
                 'Census_TotalPhysicalRAM',
                 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
                 'Census_InternalPrimaryDisplayResolutionHorizontal',
                 'Census_InternalPrimaryDisplayResolutionVertical')


# (1) NAì²˜ë¦¬?•˜ê³?, ë²”ì£¼?˜•?œ¼ë¡? ë³€?™˜?•˜ê¸?
for(i in 1:length(factor.name)){
  
  dataset.q.1[,factor.name[i]] <- NAtoFactor(dataset.q.1[,factor.name[i]])
  
}

# (2) NAë¥? ??€ì²´ê°’?œ¼ë¡? ??€ì²´í•˜ê¸?
for(i in 1:length(factor.name)){

  dataset.q.2[is.na(dataset.q.2[,factor.name[i]]),
              factor.name[i]] <- summary(dataset.q.2[,factor.name[i]])[3]

}
```

<br><br> **? „ì²˜ë¦¬?œ ?°?´?„°?…‹ 1ë²ˆì˜ NA??€ ? ˆë²¨ìˆ˜ ?™•?¸?•˜ê¸?**

``` {r}
# NAê°’ê³¼ ? ˆë²¨ìˆ˜ ?™•?¸?•˜ê¸?
factor.name <- colnames(dataset.q.1)

for(i in 1:length(factor.name)){
  
  cat('ë³€?ˆ˜ëª? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('? ˆë²¨ìˆ˜ : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

<br><br> **? „ì²˜ë¦¬?œ ?°?´?„°?…‹ 2ë²ˆì˜ NA??€ ? ˆë²¨ìˆ˜ ?™•?¸?•˜ê¸?**

``` {r}
# NAê°’ê³¼ ? ˆë²¨ìˆ˜ ?™•?¸?•˜ê¸?
factor.name <- colnames(dataset.q.2)

for(i in 1:length(factor.name)){
  
  cat('ë³€?ˆ˜ëª? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('? ˆë²¨ìˆ˜ : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

4.? „ì²˜ë¦¬?•œ ?°?´?„°?…‹?œ¼ë¡? ?˜?‚¬ê²°ì •?‚˜ë¬? ? ?•©?•˜ê¸?<br><br>
-----------------------------------------------------

**? „ì²˜ë¦¬?•œ ?°?´?„°?…‹?œ¼ë¡? ?˜?‚¬ê²°ì •?‚˜ë¬? ? ?•©?•˜ê¸? 1**

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

<br><br> **? „ì²˜ë¦¬?•œ ?°?´?„°?…‹?œ¼ë¡? ?˜?‚¬ê²°ì •?‚˜ë¬? ? ?•©?•˜ê¸? 2**

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

5. ë³€?ˆ˜ ? ˆë²? ì¶•ì†Œ?•´ë³´ê¸°
-----------------------

?¼ë¶€ ë³€?ˆ˜?“¤?˜ NAê°’ì„ Mean?œ¼ë¡? ??€ì²´í•  ?•Œ?˜ ? •?™•?„ê°€ ì¡°ê¸ˆ ?” ?†’?•˜?‹¤. ?•˜ì§€ë§?, ?•„ë¬´ëŸ° ? „ì²˜ë¦¬ë¥? ?•˜ì§€ ?•Š??€ ?ƒ?ƒœ?—?„œ ? ?•©?•œ ?˜?‚¬ê²°ì •?‚˜ë¬? ëª¨í˜•ë³´ë‹¤?Š” ë¶€ì¡±í•œ ?ˆ˜ì¹˜ë?? ë³´ì´ê³? ?žˆ?‹¤. identifier??€ ê°™ì?€ ë³€?ˆ˜?“¤?„ factorë¡? ë³€?™˜?•˜ë©´ì„œ ?ˆ˜ë§Žì?€ ? ˆë²¨ì´ ?ƒ?„±?˜?–´ ?˜¤?žˆ? ¤ ? „ì²´ì ?¸ ì§€?‘œ?“¤?˜ ?ˆ˜ì¹˜ê?€ ê°ì†Œ?•œ ê²? ê°™ë‹¤.<br><br>?”°?¼?„œ ? ˆë²¨ì˜ ?ˆ˜ë¥? ?–´?Š? •?„ ì¤„ì¸?‹¤ë©?, ?” ì¢‹ì?€ ê²°ê³¼ê°€ ?‚˜?˜¬ ê²ƒì´?¼ê³? ?Œ?‹¨?–ˆ?‹¤.<br><br>HasDetections(ëª©í‘œë³€?ˆ˜)?˜ 1ê³? 0?˜ ë¹ˆë„ê°€ ?œ ?‚¬?•œ ê²ƒë¼ë¦? ê·¸ë£¹?œ¼ë¡? ë¬¶ì–´?„œ ? ˆë²¨ì˜ ?ˆ˜ê°€ 50ê°? ?´?ƒ?¸ ì»¬ëŸ¼?„ 50ê°? ?´?•˜?˜ ? ˆë²¨ë¡œ ì¤„ì—¬ë³´ë„ë¡? ?•˜??€?‹¤.<br><br> **50ê°? ?´?ƒ?˜ ? ˆë²¨ì˜ ?ˆ˜ ì¤„ì´ê¸?**

``` {r}
factor.name <- colnames(dataset.q.1)
Com.factor.name <- c() # ? ˆë²¨ì´ 50ê°? ?´?ƒ?¸ ì»¬ëŸ¼ëª?


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.1[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.1)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.1[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.1$HasDetections, input = dataset.q.1[,Com.factor.name[i]])
  
}

# ? ˆë²? ?ˆ˜ ?‹¤?‹œ ?•œ ë²? ?™•?¸?•´ë³´ê¸°
for(i in 1:length(factor.name)){
  
  cat('ë³€?ˆ˜ëª? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.1[,factor.name[i]]), '\n')
  cat('? ˆë²¨ìˆ˜ : ', nlevels(dataset.q.1[,factor.name[i]]), 
      '\n\n')
  
}
```

``` {r}
factor.name <- colnames(dataset.q.2)
Com.factor.name <- c() # ? ˆë²¨ì´ 50ê°? ?´?ƒ?¸ ì»¬ëŸ¼ëª?


for(i in 1:length(factor.name)){
  
  if(nlevels(dataset.q.2[,factor.name[i]]) > 50){
    
    Com.factor.name <- rbind(Com.factor.name, 
                             colnames(dataset.q.2)[i])
    
  }
  
}


for(i in 1:length(Com.factor.name)){
  
  dataset.q.2[,Com.factor.name[i]] <- CompressLevels(object = dataset.q.2$HasDetections, input = dataset.q.2[,Com.factor.name[i]])
  
}

# ? ˆë²? ?ˆ˜ ?‹¤?‹œ ?•œ ë²? ?™•?¸?•´ë³´ê¸°
for(i in 1:length(factor.name)){
  
  cat('ë³€?ˆ˜ëª? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.q.2[,factor.name[i]]), '\n')
  cat('? ˆë²¨ìˆ˜ : ', nlevels(dataset.q.2[,factor.name[i]]), 
      '\n\n')
  
}
```

6. ì¶•ì†Œ?•œ ?°?´?„°?…‹?œ¼ë¡? ?˜?‚¬ê²°ì •?‚˜ë¬?, ?žœ?¤?¬? ˆ?Š¤?Š¸ ? ?•©?•˜ê¸?
----------------------------------------------------------

<br><br> ?´? œë¶€?„° ?˜?‚¬ê²°ì •?‚˜ë¬´ë?? ? ?•©?•´ë³´ê³ , ?žœ?¤?¬? ˆ?Š¤?Š¸?Š” ?Šœ?‹ê¹Œì?€ ?•´ë³´ê² ?‹¤.<br><br><br><br> **?˜?‚¬ê²°ì •?‚˜ë¬? ? ?•©?•˜ê¸? 1**

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

# ?˜¼?™?–‰? ¬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?™•?¸?š©
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# ë¹„ìš©ë³µìž¡?„ ?‘œ ì¶œë ¥
printcp(fitTree)
```

**?˜?‚¬ê²°ì •?‚˜ë¬? ? ?•©?•˜ê¸? 2**

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

# ?˜¼?™?–‰? ¬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?™•?¸?š©
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)

# ë¹„ìš©ë³µìž¡?„ ?‘œ ì¶œë ¥
printcp(fitTree)
```

<br><br><br> ? ˆë²¨ì˜ ?ˆ˜ë¥? ì¶•ì†Œ?•œ ?´?›„ë¡œëŠ” ëª¨ë“  ì»¬ëŸ¼?„ ë²”ì£¼?˜•?œ¼ë¡? ë³€ê²½í•œ ?°?´?„°?…‹?—?„œ ?” ?†’??€ F1 ? ?ˆ˜??€ auroc ê°’ì´ ?‚˜?™”?‹¤. ?ƒ?™©?´ ?—­? „?˜?—ˆ?‹¤. ê°ê° ëª¨í˜•?—?„œ ê°€?ž¥ ë§ˆì?€ë§? ë¶„ë¦¬?—?„œ xerrorê°€ ê°€?ž¥ ?‚®ê²? ?‚˜?™”?œ¼ë¯€ë¡?, ê°€ì§€ì¹˜ê¸°ë¥? ?•  ?•„?š”ê°€ ?—†?—ˆ?‹¤. ê·¸ë ‡?‹¤ë©? ?´? œ ëª¨ë“  ì»¬ëŸ¼?„ ë²”ì£¼?˜•?œ¼ë¡? ë³€ê²½í•œ ?°?´?„°?…‹?œ¼ë¡? ?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜•?„ ? ?•©?•´ë³´ê¸°ë¡? ?•˜??€?‹¤.<br><br> **?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜• ê°„ë‹¨?•˜ê²? ? ?•©?•´ë³´ê¸°**

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

# ëª¨í˜• ? ?•© ê²°ê³¼ ?™•?¸?•˜ê¸? ( ?˜¤ë¶„ë¥˜?œ¨ ?™•?¸?•˜ê¸? )
print(fitRFC)

# ë³€?ˆ˜ ì¤‘ìš”?„ ì¶œë ¥?•˜ê¸?
importance(fitRFC)

# ë§ˆì§„ ê·¸ëž˜?”„
plot(margin(fitRFC))

# ?˜¼?™?–‰? ¬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?™•?¸?š©
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> MeanDecreaseAccuracy?—?„œ ?¬ê²? ?˜?–¥?„ ì£¼ëŠ” ë³€?ˆ˜?Š”,<br> AppVersion 13.2078028<br> AVProductStatesIdentifier 21.1545553<br> CountryIdentifier 10.5903571<br> CityIdentifier 51.7872658<br> SmartScreen 31.7732623<br> Census\_OEMModelIdentifier 40.9204674<br> Census\_ProcessorModelIdentifier 13.0899241<br> Census\_SystemVolumeTotalCapacity 103.9377516<br> Census\_InternalBatteryNumberOfCharges 16.4877933<br> Census\_OSInstallLanguageIdentifier 13.7160384<br> Census\_FirmwareVersionIdentifier 31.2409193<br> Wdft\_RegionIdentifier 12.5699366<br><br> ?´ê²ƒë“¤?´?‹¤. ê°? ë³€?ˆ˜?“¤?˜ ?˜ë¯¸ë?? ê°„ë‹¨?•˜ê²? ?•Œ?•„ë³´ìžë©?,<br> 1. windows defender?˜ ë²„ì „<br> 2. ?•ˆ?‹° ë°”ì´?Ÿ¬?Š¤ ë°±ì‹ ?˜ ë²„ì „<br> 3. êµ?ê°€ì½”ë“œ, ?„?‹œì½”ë“œ, ì§€?—­ì½”ë“œ, ?–¸?–´ì½”ë“œ<br> 4. smartscreen(?œˆ?„?š° 10?˜ ë°©í™”ë²? ê¸°ëŠ¥)<br> 5. OEM?˜ ì¢…ë¥˜(?š´?˜ì²´ì œë¥? ??€?Ÿ‰?œ¼ë¡? êµ¬ë§¤?•˜?—¬ ?„¤ì¹˜í•˜?Š” ë°©ì‹)<br> 6. CPU?˜ ëª¨ë¸ ëª?<br> 7. ?š´?˜ì²´ì œê°€ ê¹”ë ¤?žˆ?Š” ?ŒŒ?‹°?…˜?˜ ?¬ê¸?<br> 8. ë°©í™”ë²½ì˜ ë²„ì „<br> ?´?‹¤. ê·¸ë¦¬ê³? ?˜¼?™?–‰? ¬?„ ?™•?¸?•´ë³? ê²°ê³¼, ?´ ëª¨í˜•??€ ê°ì—¼?˜ì§€ ?•Š??€ PCë¥? ?” ?ž˜ ì°¾ì•„?‚´?Š” ?Š¹?´?„??€ ? •ë°€?„ê°€ ?†’??€ ëª¨í˜•?´?‹¤.<br> ?´ ëª¨í˜•??€ ? •ë°€?„ê°€ ?ƒ?‹¹?žˆ ?†’??€ ?Ž¸?´ì§€ë§?, ?•…?„± ì½”ë“œ?— ê°ì—¼?œ ì»´í“¨?„°ë¥? ê°ì—¼?˜ì§€ ?•Š?•˜?‹¤ê³? ?Œ?‹¨?•˜?Š” ê²½ìš°ê°€ ?ƒ?‹¹?•˜ê¸? ?•Œë¬¸ì— ?•„?‰¬?š´ ëª¨í˜•?´?‹¤.<br> ê·¸ë ‡ê¸? ?•Œë¬¸ì— ?Šœ?‹?„ ?•œ?‹¤ë©?, ë¯¼ê°?„ê°€ ?” ?†’??€ ëª¨í˜•?´ ?‚˜?˜¬ ?ˆ˜ ?žˆì§€ ?•Š?„ê¹? ?ƒê°ì„ ?•´ë³´ì•˜?‹¤.<br> **?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜• ?Šœ?‹?•´ë³´ê¸°**

``` {r}
# # ì´? 32ê°œì˜ ì¡°í•©
# # ??€ ?´?Ÿ° ì¢‹ì?€ ?•¨?ˆ˜ê°€ ?žˆ?—ˆ?–´?
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
#   # ?˜ˆì¸¡ê°’
#   trPred <- fit$test$predicted
#   # ?‹¤? œê°?
#   trReal <- dataset.q.1.v$HasDetections
#   # ?˜¼?™?–‰? ¬
#   con <- confusionMatrix(trPred, trReal, positive = '1')
#   
#   # ?˜¤ë¶„ë¥˜?ˆ˜
#   mcSum <- sum(fit$predicted != dataset.q.1.t$HasDetections)
#   # ?˜¤ë¶„ë¥˜?œ¨
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

<br><br> ?‹œê°„ì´ ?—†?Š” ê´€ê³„ë¡œ ë§¤ìš° ê°„ë‹¨?•˜ê²? ê·¸ë¦¬?“œ ?„œì¹˜ë?? ?•˜??€?‹¤.<br> ntree = 500, mtry = 13? •?„ë¡? ?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜•?„ ? ?•©?•˜ë©? ? ?‹¹?•  ê²? ê°™ë‹¤.<br><br> **?Šœ?‹?•œ ?žœ?¤?¬? ˆ?Š¤?Š¸ ëª¨í˜•**

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

# ëª¨í˜• ? ?•© ê²°ê³¼ ?™•?¸?•˜ê¸? ( ?˜¤ë¶„ë¥˜?œ¨ ?™•?¸?•˜ê¸? )
print(fitRFC)

# ë³€?ˆ˜ ì¤‘ìš”?„ ì¶œë ¥?•˜ê¸?
importance(fitRFC)

# ë§ˆì§„ ê·¸ëž˜?”„
plot(margin(fitRFC))

# ?˜¼?™?–‰? ¬
confusionMatrix(trPred, trReal, positive = '1')

# F1_Score
F1_Score(trPred, trReal)

# auroc ?™•?¸?š©
Pred <- trPred %>% as.numeric()
Real <- trReal %>% as.numeric()

# auroc
auc(Real, Pred)
```

7. ë²ˆì™¸ - XGBoost ?‚¬?š©?•´ë³´ê¸°<br><br>
------------------------------------

ê°ì¢… ??€?šŒ?—?„œ ?†’??€ ? ?ˆ˜ë¥? ê¸°ë¡?•˜?Š” ë¨¸ì‹ ?Ÿ¬?‹ ?•Œê³ ë¦¬ì¦? ì¤? ?•˜?‚˜?¸, XGBoostë¥? ?‚¬?š©?•˜?—¬ ?„±?Š¥?„ ?‰ê°€?•´ë³´ê¸°ë¡? ?•˜??€?‹¤.<br> **?¼ë²¨ë§, ?”ë¯¸ë?€?ˆ˜ ?•¨?ˆ˜**

``` {r}
# ?¼ë²¨ë§ ?•¨?ˆ˜ ( numeric vectorë¡? ë°˜í™˜ )
MakingLabel <- function(data.variable){
  
  for(i in 1:nlevels(data.variable)){
  
    levels(data.variable)[i] <- i
  
  }
  
  data.variable <- as.numeric(data.variable)
  return(data.variable)
}

# ?”ë¯¸ë?€?ˆ˜ ë§Œë“œ?Š” ?•¨?ˆ˜ ( ?”ë¯¸ë?€?ˆ˜?“¤?„ data.frame?œ¼ë¡? ë°˜í™˜ )
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

# ?‚¬?š©?˜ˆ?‹œ
example.label <- MakingLabel(dataset.q.1$ProductName)
example.dummy <- MakingDummy(dataset.q.1$ProductName,
                             'result')
```

<br><br> **?°?´?„°?…‹ ë³€?™˜?•˜ê¸?**

``` {r}
# ?°?´?„°?…‹
dataset.n <- dataset.q.1

HasDetections <- dataset.n$HasDetections %>% as.numeric()
HasDetections <- ifelse(HasDetections == 2,0,1)

dataset.n.label <- dataset.n[, -82]
dataset.n.dummy <- dataset.n

# dataset.n.dummy?˜ ? ˆë²? ?ˆ˜ ?žë¥´ê¸°
factor.name <- colnames(dataset.n.dummy)
Com.factor.name <- c() # ? ˆë²¨ì´ 50ê°? ?´?ƒ?¸ ì»¬ëŸ¼ëª?


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

# ? ˆë²? ?ˆ˜ ?‹¤?‹œ ?™•?¸?•˜ê¸?
for(i in 1:length(factor.name)){
  
  cat('ë³€?ˆ˜ëª? : ', factor.name[i], "\n")
  cat('NA : ', naniar::n_miss(dataset.n.dummy[,factor.name[i]]), '\n')
  cat('? ˆë²¨ìˆ˜ : ', nlevels(dataset.n.dummy[,factor.name[i]]), 
      '\n\n')
  
}

# ëª©í‘œë³€?ˆ˜ë§? ? œê±°í•˜ê¸?
dataset.n.dummy <- dataset.n.dummy[, -82]

# factor.name?—?„œ ëª©í‘œë³€?ˆ˜ ? œê±°í•˜ê¸?
factor.name <- colnames(dataset.n)[-82]

# 1. ?¼ë²¨ë§?•˜ê¸?
for(i in 1:length(factor.name)){
  
  dataset.n.label[, factor.name[i]] <- MakingLabel(dataset.n.label[, factor.name[i]])
  
}

dataset.n.label <- cbind(dataset.n.label, HasDetections)
ncol(dataset.n.label)

# 2. ?”ë¯¸ë?€?ˆ˜?™”
for(i in 1:length(factor.name)){
  
  dataset.n.dummy <- cbind(dataset.n.dummy,
                           MakingDummy(dataset.n.dummy[,factor.name[i]],
                                       factor.name[i]))
  
}

dataset.n.dummy <- dataset.n.dummy[,-c(1:81)]
dataset.n.dummy <- cbind(dataset.n.dummy, HasDetections)
ncol(dataset.n.dummy)

# trainsetê³? validationset?œ¼ë¡? ?‚˜?ˆ„ê¸?
dataset.n.label.t <- dataset.n.label[index == 1, ]
dataset.n.dummy.t <- dataset.n.dummy[index == 1, ]

dataset.n.label.v <- dataset.n.label[index == 2, ]
dataset.n.dummy.v <- dataset.n.dummy[index == 2, ]
```

<br><br> **XGBoost ?‚¬?š©?•´ë³´ê¸° 1**

``` {r}
# 1. ?¼ë²¨ë§?•œ ?°?´?„°?…‹
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.label.t[ ,-82]),
                      label= as.matrix(dataset.n.label.t[ , 82]))


# ?ŒŒ?¼ë¯¸í„°
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds ì°¾ê¸°
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost ëª¨í˜• ? ?•©?•˜ê¸?
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.label.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.label.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# ?˜¼?™?–‰? ¬
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc ?™•?¸?š©
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

<br><br> **XGBoost ?‚¬?š©?•´ë³´ê¸° 2**

``` {r}
# 2. ?”ë¯¸ë?€?ˆ˜ë¡? ë§Œë“  ?°?´?„°?…‹
dtrain <- xgb.DMatrix(data = as.matrix(dataset.n.dummy.t[ ,-82]),
                      label= as.matrix(dataset.n.dummy.t[ , 82]))


# ?ŒŒ?¼ë¯¸í„°
default_param<-list(
  objective = 'binary:logistic',
  booster = 'gbtree',
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  colsample_bytree=1
)

# nrounds ì°¾ê¸°
xgbcv <- xgb.cv(params = default_param,
                data = dtrain,
                nrounds = 200,
                nfold = 10,
                verbose = 1,
                print_every_n = 25,
                early_stopping_rounds = 20)

# nrounds
xgbcv$best_iteration

# xgboost ëª¨í˜• ? ?•©?•˜ê¸?
newxgb <- xgboost(params = default_param,
                  verbose = 1,
                  data = dtrain,
                  nrounds = xgbcv$best_iteration,
                  print_every_n = 25)


pred <- predict(newxgb, as.matrix(dataset.n.dummy.v[, -82]))
pred <- ifelse(pred > 0.5, 1, 0)
pred <- as.factor(pred) %>% relevel(ref = '1')

real <- dataset.n.dummy.v[, 82] %>% as.factor() %>% relevel(ref = '1')

# ?˜¼?™?–‰? ¬
confusionMatrix(pred, real, positive = '1')

# F1_Score
F1_Score(pred, real)

# auroc ?™•?¸?š©
Pred <- pred %>% as.numeric()
Real <- real %>% as.numeric()

# auroc
auc(Real, Pred)
```

#### ?<br><br>

8. ë§ˆë¬´ë¦?
---------

?š°ë¦¬ë“¤?˜ ê´€?‹¬ë¶„ì•¼?¸ ?•…?„±ì½”ë“œ?— ??€?•œ ?˜ˆì¸? ?°?´?„°?…‹?— ??€?•´?„œ?Š” ?–´?Š? •?„ ê°œìš”ê°€ ?ž¡?žŒ ?ƒ?ƒœ??€ì§€ë§?, ?Šœ?‹ê³? ? „ì²˜ë¦¬?— ?žˆ?–´?„œ ?—¬?Ÿ¬ê°€ì§€ ?•„?‰¬?š´ ?•œê³„ì ?“¤?´ ?žˆ?—ˆ?‹¤.<br> 1. 1500ë§Œí–‰?˜ ?‹¤? œ ?°?´?„°?—?„œ?Š” testset?—?Š” ?žˆì§€ë§?, trainset?—?Š” ?—†?Š” Identifierê°€ ?žˆê¸? ?•Œë¬¸ì—, ?•½ 5~10%? •?„?˜ ? •ë³´ë?? ?‚¬?š©?•˜ì§€ ëª»í•˜?Š” ê²ƒì„ ?™•?¸?–ˆ?‹¤.<br><br> 2. ?‹¤? œ ?°?´?„°?—?„œ?Š” ?˜„?ž¬?˜ ë°©ë²• ì¤? ëª‡êµ°??€ë¥? ?ˆ˜? •?•´?•¼?•˜ê³?, ê·¸ë¡œ ?¸?•´?„œ ì¹˜ëª…? ?¸ ?˜¤ì°¨ê?€ ë°œìƒ?•  ?ˆ˜?„ ?žˆ?‹¤.<br><br> 3. ì»´í“¨?„°?˜ ?‚¬?–‘ ë¬¸ì œë¡? ?‹¤? œ ?°?´?„°?Š” ? „ì²˜ë¦¬ ?‹œê°„ì—ë§? 7?¼ ?´?ƒ?´ ?†Œ?š”?  ê²ƒì´?‹¤.<br><br> 4. ê·¸ë ‡ê¸? ?•Œë¬¸ì—, ?‹¤? œ ?°?´?„°ë¥? ?‹¤ë¤„ë³´ì§€ ëª»í•˜??€ê³?, ?‹¤?–‘?•œ ë°©ë²•?“¤?„ ê³ ë?¼í•˜ê¸°ì—?Š” ?•œê³„ê?€ ?žˆ?—ˆ?‹¤. ?´ ê³¼ì •?´ ??‚œ ?›„?— ê°œì¸? ?œ¼ë¡? ?´ ??€?šŒë¥? ë§ˆë¬´ë¦? ì§€?„ ?ƒê°ì´?‹¤.
