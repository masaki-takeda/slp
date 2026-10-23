library(tidyr)
library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)

## PATH
path_data = "~/LMM/Data_5"
setwd(path_data)

## DATA READ
dst = read.csv("dst_score.csv")
pupil = read.csv("pupil_size.csv")
eeg = read.csv("eeg.csv")

## MERGE
names(dst)[2:91] = 1:90
dst_long <- dst %>%
  pivot_longer(
    cols = `1`:`90`,
    names_to = "Trial",
    values_to = "DST"
  ) %>%
  rename(Subject = Sub) %>%
  mutate(
    Trial = as.numeric(Trial)
  )

eeg_merged <- eeg %>%
  left_join(
    dst_long,
    by = c("Subject", "Trial")
  )

eeg_merged <- eeg_merged %>%
  select(-StageNam)

pupil_long <- pupil %>%
  pivot_longer(
    cols = starts_with("trial"),
    names_to = "Trial",
    values_to = "Pupil"
  ) %>%
  rename(Subject = X) %>%
  mutate(
    Subject = sprintf(
      "Sub%02d",
      as.numeric(gsub("Sub", "", Subject))
    ),
    Trial = as.numeric(gsub("trial", "", Trial))
  )

eeg_merged <- eeg_merged %>%
  left_join(
    pupil_long %>% select(Subject, Trial, Pupil),
    by = c("Subject", "Trial")
  )

## =========================
## Settings
## =========================

bands   <- unique(eeg_merged$Band)
regions <- unique(eeg_merged$Region)

z_score <- 1

## for saving
result_group_each <- data.frame()
result_interaction <- data.frame()


## =========================
## 1. LMM per Group
## DST ~ ERSP * Trial + Pupil
## =========================

for(g in unique(eeg_merged$Group)){
  
  for(b in bands){
    
    for(r in regions){
      
      dat <- subset(
        eeg_merged,
        Group == g & Band == b & Region == r
      )
      
      if(z_score == 1){
        dat$ERSP_Value <- as.numeric(scale(dat$ERSP_Value))
        dat$Pupil      <- as.numeric(scale(dat$Pupil))
        dat$Trial      <- as.numeric(scale(dat$Trial))
      }
      
      fit <- lmer(
        DST ~ ERSP_Value * Trial + Pupil +
          (1 + Trial || Subject),
        data = dat,
        control = lmerControl(
          optimizer = "bobyqa",
          optCtrl = list(maxfun = 100000)
        )
      )
      
      coef_tab <- summary(fit)$coefficients
      
      result_group_each <- rbind(
        result_group_each,
        data.frame(
          Group  = g,
          Band   = b,
          Region = r,
          
          Beta_ERSP = coef_tab["ERSP_Value", "Estimate"],
          SE_ERSP   = coef_tab["ERSP_Value", "Std. Error"],
          t_ERSP    = coef_tab["ERSP_Value", "t value"],
          p_ERSP    = coef_tab["ERSP_Value", "Pr(>|t|)"],
          
          Beta_ERSP_Trial = coef_tab["ERSP_Value:Trial", "Estimate"],
          SE_ERSP_Trial   = coef_tab["ERSP_Value:Trial", "Std. Error"],
          t_ERSP_Trial    = coef_tab["ERSP_Value:Trial", "t value"],
          p_ERSP_Trial    = coef_tab["ERSP_Value:Trial", "Pr(>|t|)"],
          
          Beta_Pupil = coef_tab["Pupil", "Estimate"],
          p_Pupil    = coef_tab["Pupil", "Pr(>|t|)"]
        )
      )
    }
  }
}

## FDR correction
result_group_each <- result_group_each %>%
  group_by(Group) %>%
  mutate(
    p_ERSP_FDR = p.adjust(p_ERSP, method = "fdr"),
    p_ERSP_Trial_FDR = p.adjust(p_ERSP_Trial, method = "fdr")
  ) %>%
  ungroup()


## =========================
## 2. interaction model
## DST ~ ERSP * Group * Trial + Pupil
## =========================

for(b in bands){
  
  for(r in regions){
    
    dat <- subset(
      eeg_merged,
      Band == b & Region == r
    )
    
    if(z_score == 1){
      dat$ERSP_Value <- as.numeric(scale(dat$ERSP_Value))
      dat$Pupil      <- as.numeric(scale(dat$Pupil))
      dat$Trial      <- as.numeric(scale(dat$Trial))
    }
    
    fit <- lmer(
      DST ~ ERSP_Value * Group * Trial + Pupil +
        (1 + Trial || Subject),
      data = dat,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 100000)
      )
    )
    
    coef_tab <- summary(fit)$coefficients
    
    result_interaction <- rbind(
      result_interaction,
      data.frame(
        Band   = b,
        Region = r,
        
        Beta_ERSP_Group =
          coef_tab["ERSP_Value:GroupShallow-NREM", "Estimate"],
        SE_ERSP_Group =
          coef_tab["ERSP_Value:GroupShallow-NREM", "Std. Error"],
        t_ERSP_Group =
          coef_tab["ERSP_Value:GroupShallow-NREM", "t value"],
        p_ERSP_Group =
          coef_tab["ERSP_Value:GroupShallow-NREM", "Pr(>|t|)"],
        
        Beta_ERSP_Group_Trial =
          coef_tab["ERSP_Value:GroupShallow-NREM:Trial", "Estimate"],
        SE_ERSP_Group_Trial =
          coef_tab["ERSP_Value:GroupShallow-NREM:Trial", "Std. Error"],
        t_ERSP_Group_Trial =
          coef_tab["ERSP_Value:GroupShallow-NREM:Trial", "t value"],
        p_ERSP_Group_Trial =
          coef_tab["ERSP_Value:GroupShallow-NREM:Trial", "Pr(>|t|)"],
        
        Beta_Pupil = coef_tab["Pupil", "Estimate"],
        p_Pupil    = coef_tab["Pupil", "Pr(>|t|)"]
      )
    )
  }
}

## FDR correction
result_interaction <- result_interaction %>%
  mutate(
    p_ERSP_Group_FDR =
      p.adjust(p_ERSP_Group, method = "fdr"),
    p_ERSP_Group_Trial_FDR =
      p.adjust(p_ERSP_Group_Trial, method = "fdr")
  )


## =========================
## 3. mark significance
## =========================

result_group_each <- result_group_each %>%
  mutate(
    sig_ERSP = case_when(
      p_ERSP_FDR < .001 ~ "***",
      p_ERSP_FDR < .01  ~ "**",
      p_ERSP_FDR < .05  ~ "*",
      TRUE ~ ""
    ),
    sig_ERSP_Trial = case_when(
      p_ERSP_Trial_FDR < .001 ~ "***",
      p_ERSP_Trial_FDR < .01  ~ "**",
      p_ERSP_Trial_FDR < .05  ~ "*",
      TRUE ~ ""
    )
  )

result_interaction <- result_interaction %>%
  mutate(
    sig_ERSP_Group = case_when(
      p_ERSP_Group_FDR < .001 ~ "***",
      p_ERSP_Group_FDR < .01  ~ "**",
      p_ERSP_Group_FDR < .05  ~ "*",
      TRUE ~ ""
    ),
    sig_ERSP_Group_Trial = case_when(
      p_ERSP_Group_Trial_FDR < .001 ~ "***",
      p_ERSP_Group_Trial_FDR < .01  ~ "**",
      p_ERSP_Group_Trial_FDR < .05  ~ "*",
      TRUE ~ ""
    )
  )


## =========================
## 4. plot main effects
## =========================

ggplot(result_group_each,
       aes(x = Region, y = Band, fill = Beta_ERSP)) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = paste0(round(Beta_ERSP, 3), sig_ERSP)),
    size = 5
  ) +
  facet_wrap(~ Group) +
  scale_y_discrete(
    limits = rev(c("delta", "theta", "alpha", "beta1", "beta2"))
  ) +
  scale_x_discrete(
    limits = c("frontal", "parietal", "temporal", "occipital")
  ) +
  scale_fill_gradient2(
    low = "#005aff",
    mid = "white",
    high = "#ff4b00",
    midpoint = 0
  ) +
  labs(
    x = "Region",
    y = "Band",
    fill = "β ERSP"
  ) +
  theme_classic(base_size = 18)

## =========================
## 5. plot interactions
## =========================

ggplot(result_interaction,
       aes(x = Region, y = Band, fill = Beta_ERSP_Group)) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = paste0(round(Beta_ERSP_Group, 3), sig_ERSP_Group)),
    size = 5
  ) +
  scale_y_discrete(
    limits = rev(c("delta", "theta", "alpha", "beta1", "beta2"))
  ) +
  scale_x_discrete(
    limits = c("frontal", "parietal", "temporal", "occipital")
  ) +
  scale_fill_gradient2(
    low = "#005aff",
    mid = "white",
    high = "#ff4b00",
    midpoint = 0
  ) +
  labs(
    x = "Region",
    y = "Band",
    fill = "β ERSP × Group"
  ) +
  theme_classic(base_size = 18)
