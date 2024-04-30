library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(car)
library(corrplot)

# Load the data
df_los <- read_csv('./data/los_cleaned.csv')

# Trim spaces in column names
colnames(df_los) <- trimws(colnames(df_los))
print(colnames(df_los))

# Display first few rows
head(df_los)

# Initial transformations and calculations
df_los <- df_los %>%
  mutate(LOSDiscrepancyCost = -(LOS - `GM-LOS`) * 1000) %>%
  filter(riskOfMortality != '0-Ungroupable', !is.na(LOSDiscrepancyCost)) %>%
  mutate(principalProcedure = na_if(principalProcedure, '-')) %>%
  drop_na()

# Check for multicollinearity in numeric variables
numeric_vars <- df_los %>% select_if(is.numeric)
cor_matrix <- cor(na.omit(numeric_vars))
corrplot(cor_matrix, method = "color")

# Preparing data with reference categories for factors
regress_me <- df_los %>%
  select(patientID, admitType, admittedDOW, ageGroup, race, severity, LOSDiscrepancyCost) %>%
  mutate(
    admitType = factor(admitType, levels = c("Elective", "Emergency", "Urgent")),
    admittedDOW = factor(admittedDOW, levels = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")),
    ageGroup = factor(ageGroup, levels = c("0-18", "19-35", "36-55", "56-75", "76+")),
    race = factor(race, levels = c("White", "Black", "Asian", "Other")),
    severity = factor(severity, levels = c("Minor", "Moderate", "Major", "Extreme"))
  ) %>%
  drop_na()

# Diagnostic checks
print(levels(regress_me$admitType))
print(levels(regress_me$admittedDOW))
print(levels(regress_me$ageGroup))
print(levels(regress_me$race))
print(levels(regress_me$severity))
print(nrow(regress_me))

# Inspect data consistency post-filtering
table(df_los$admittedDOW)
table(df_los$ageGroup)
table(df_los$race)
table(df_los$severity)

# Check if there is any data left to model
if(nrow(regress_me) > 0) {
  # Fit the simplified linear model
  simple_model <- lm(LOSDiscrepancyCost ~ admitType + admittedDOW + ageGroup + race + severity, data = regress_me)
  summary(simple_model)
  
  # Check for multicollinearity with VIF
  vif_values <- vif(simple_model)
  print(vif_values)
} else {
  print("No data available to fit the model. Please adjust your filtering criteria.")
}

