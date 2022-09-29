if (!require(c("frm", "tidyverse","lubridate","fastDummies","stringr")))
  install.packages(c("frm", "tidyverse","lubridate","fastDummies","stringr"))


# Load the libraries
library(frm)
library(tidyverse)
library(lubridate)
library(fastDummies)
library(stringr)


d0 <- read.csv("asia_merged.csv")
names(d0)
unique(d0$lang)
table(d0$lang)

# Checks / Explore
summary(d0)
n_distinct(d0$locationId)
anyNA(d0$locationId)
summary(d0$functionality) # 10819 NAs

#### Pre-processing + dummies ####
## Plug types + Num connectors
unique(d0$plugTypes)
d0$num_connectors <- NA
d0$num_connectors<-str_count(d0$plugTypes, pattern = ",")+1
summary(d0$num_connectors)

## Networks
unique(d0$networks)
network_tab <- d0 %>%
  group_by(networks) %>%
  summarise(
    count=n()
  )
network_tab

d0$network_grp <- "Other"
d0$network_grp[is.na(d0$networks)] <- 'None'
d0$network_grp[d0$networks == 'Tesla Destination'] <- 'Tesla'
d0$network_grp[d0$networks == 'Supercharger'] <- 'Tesla'
d0$network_grp[d0$networks == 'Tesla Destination, Supercharger'] <- 'Tesla'
d0$network_grp[d0$networks == 'Supercharger, Tesla Destination'] <- 'Tesla'
d0$network_grp[d0$networks == 'ChargePoint'] <- 'ChargePoint'
d0$network_grp[d0$networks == 'Greenlots'] <- 'Greenlots'
anyNA(d0$network_grp)

### add a dummy for network versus non-network
d0$network_du <- 1
d0$network_du[is.na(d0$networks)] <- 0

### add a dummy for government versus non-government
d0$gov <- 0
d0$gov[d0$poi == 'government'] <- 1
d0$gov[d0$poi == 'transit_station'] <- 1

### add a dummy for Taiwan or not
d0$tw <- 0
d0$tw[d0$nation == 'Taiwan'] <- 1


unique(d0$network_grp)
network_grp_tab <- d0 %>%
  group_by(network_grp) %>%
  summarise(
    count=n()
  )
network_grp_tab

## Has comment dummy
d0$has_comment <- ifelse(is.na(d0$comment),0,1)
# View(d0[,c("comment", "has_comment")])
summary(d0$has_comment)

## Extract year
d0$dates <- mdy(d0$date)
# View(d0[,c("date", "dates")])
d0$year <- year(d0$dates)
# View(d0[,c("date", "dates", "year")])

## Create English dummy and replace with 0 if NA
d0$english_dummy <- d0$english
d0[is.na(d0$english),]$english_dummy<-0

## Country/territory dummies
unique(d0$nation)
d1 <- dummy_cols(d0, select_columns =  "nation")
names(d1)

## Year dummies
d2 <- dummy_cols(d1, select_columns =  "year")

## Network dummies
d2 <- dummy_cols(d2, select_columns =  "network_grp")

## Treat var names with space or slash -> period
colnames(d2) <- gsub(" |/", '.', colnames(d2))

## Group HK+Macao, drop Indonesia
d2 <- subset(
  d2,
  nation_Indonesia!=1
)
d2$nation_Hong.Kong = d2$nation_Hong.Kong + d2$nation_Macao
names(d2)[names(d2) == 'nation_Hong.Kong'] <- 'nation_HK.Macao'
d2 <- select(d2, -c(nation_Macao, nation_Indonesia))

## POIs: Group shopping_center, supermarket, dining into retail
d2$poi_retail = d2$poi_retail + d2$poi_shopping_center + d2$poi_supermarket + d2$poi_dining
d2 <- select(d2, -c(poi_shopping_center, poi_supermarket, poi_dining))

## POIs: Group government and transit into government
d2$poi_government = d2$poi_government + d2$poi_transit_station
d2 <- select(d2, -c(poi_transit_station))

## Dummy lists
dl_nation<-colnames(d2)[grepl("nation_",colnames(d2))]
dl_nation

# rename poi_other/unknown
# rename(d2, poi_other = poi_other/unknown)
# summary(d2$poi_other)

dl_poi<-colnames(d2)[grepl("poi_",colnames(d2))]
dl_poi

dl_year<-colnames(d2)[grepl("year_",colnames(d2))]
dl_year

dl_network<-colnames(d2)[grepl("network_grp_",colnames(d2))]
dl_network


## Summaries
sum_country<- d0 %>% 
  group_by(nation)%>% 
  summarise(
    count=n()
  )

sum_country_yr<- d0 %>% 
  group_by(nation,year)%>% 
  summarise(
    count=n()
  )

sum_poi<- d0 %>% 
  group_by(poi)%>% 
  summarise(
    count=n()
  )

sum_lang<- d0 %>% 
  group_by(lang)%>% 
  summarise(
    count=n()
  )

sum_country_review<- d0 %>% 
  group_by(nation,has_comment)%>% 
  summarise(
    count=n()
  )

sum_country_lang<- d0 %>% 
  group_by(lang,nation)%>% 
  summarise(
    count=n()
  )
View(sum_country_lang)
View(sum_country_review)
# View(sum_poi)

#### Station-year scores ####

# Testing/validation
locations_scores <- d0 %>%
  group_by(locationId, year) %>%
  mutate(count = n()) %>%
  mutate(success_ratio= sum(isSuccess)/count)
arrange(d0, locationId, year)
View(locations_scores[,c("locationId", "year", "count", "isSuccess", "success_ratio")])
# lgtm

locations_scores <- d2 %>%
  group_by(locationId, year) %>%
  mutate(count = n()) %>%
  mutate(success_rate= sum(isSuccess)/count) %>%
  mutate(error_rate= sum(isError)/count) %>%
  mutate(english_ratio = sum(english_dummy)/count) %>%
  mutate(all_english = ifelse(english_ratio==1,1,0)) %>%
  mutate(total_comments = sum(has_comment))

d3 <- cbind(d2, locations_scores[,c("count",
                                    "success_rate",
                                    "error_rate", 
                                    "english_ratio", 
                                    "all_english", 
                                    "total_comments"
)])

# View(d3[,c("locationId", "year", "count", "isSuccess", "success_rate", "isError", "error_rate", "english_ratio", "all_english",
#            colnames(d3)[grep("nation",colnames(d3))],
#            colnames(d3)[grep("year",colnames(d3))]
#            )])

# View(d3[,c("locationId", "year", "count", "total_comments", "functionality", "functionality_ratio")])

## Translated dummy var
d3$translated <- 1-d3$all_english

# View(d2[,colnames(d2)[grep("nation",colnames(d2))]])

#### Econometric Models ########################

# Regressors: translated
# Categorical dummies:
# Nation (base: China)
# Year (base: 2011)
# POI (base: dining)

###### Models with all points of interest as seperate variables
# Model 0 - Error rate

x_regressors <- c("num_connectors","poi_government","poi_retail","poi_lodging","poi_services","poi_entertainment","poi_gas_station","network_du")

x0 <- subset(d3, select = c(x_regressors, dl_year, dl_nation))
# x0 <- subset(d3, select = c(x_regressors, dl_poi, dl_nation))
# x0 <- subset(d3, select = c(x_regressors, dl_nation))

x0 <- subset(x0, select = -c(nation_China,
                             
                             year_2011
)
)

m0 <- frm(d3$error_rate, x0 , linkfrac = 'logit', var.type =  "cluster", var.cluster =d3$locationId)
sink('frm_m01.txt')
m0r <- frm.pe(m0)
sink()
# write.csv(m0r,'frm_m01.csv')


# Model 1 - Functionality ratio - Reviews only

d4 <- d3 %>%
  filter(!is.na(d3$functionality))

## Drop 2011 reviews - only 3
d4 <- subset(
  d4,
  year_2011!=1
)
# d4 <- select(d4, -c(year_2011))

d4 <- d4 %>%
  group_by(locationId, year) %>%
  mutate(count = n()) %>%
  mutate(functionality_ratio = sum(functionality)/count)

## Checks
# summary(d4$functionality_ratio)
# anyNA(d4$functionality_ratio)

x1 <- subset(d4, select = c(x_regressors, dl_network, dl_nation, dl_year))
x1 <- subset(x1, select = -c(nation_China,
                            
                             year_2011,
                             year_2012,
                             year_2013,
                             year_2014))

# x1 <- subset(x1, select = -c(nation_Indonesia))
# x1 <- subset(x1, select = -c(year_2012, year_2013))

# summary(x1)
d4 %>% 
  group_by(year)%>% 
  summarise(
    count=n()
  )

y1 <- d4$functionality_ratio

m1 <- frm(y1, x1, linkfrac = 'logit', var.type =  "cluster", var.cluster =d4$locationId)

sink('frm_m11.txt')
m1r <- frm.pe(m1)
sink()
write.csv(m1r,'frm_m11.csv')

##############################################
### Models adding "translated" variable
### Model 0.1 - Error rate
names(d3)
x_regressors <- c("translated","num_connectors","poi_government","poi_retail","poi_lodging","poi_services","poi_entertainment","poi_gas_station","network_du")

x0 <- subset(d3, select = c(x_regressors, dl_year, dl_nation))
# x0 <- subset(d3, select = c(x_regressors, dl_poi, dl_nation))
# x0 <- subset(d3, select = c(x_regressors, dl_nation))

x0 <- subset(x0, select = -c(nation_China,
                             
                             year_2011
)
)

m0 <- frm(d3$error_rate, x0 , linkfrac = 'logit', var.type =  "cluster", var.cluster =d3$locationId)
sink('frm_m21.txt')
m0r <- frm.pe(m0)
sink()
write.csv(m0r,'frm_m21.csv')

names(d3)
# model01<-lm(d3$error_rate~d3$translated+d3$num_connectors+d3$gov+d3$network_du+d3$nation_HK.Macao+d3$nation_Japan+d3$nation_Malaysia+d3$nation_Philippines+d3$nation_Singapore+d3$nation_South.Korea+d3$nation_Taiwan+d3$nation_Thailand+d3$year_2012+d3$year_2013+d3$year_2014+d3$year_2015+d3$year_2016+d3$year_2017+d3$year_2018+d3$year_2019+d3$year_2020+d3$year_2021)
# summary(model01)

# Model 1 - Functionality ratio - Reviews only

d4 <- d3 %>%
  filter(!is.na(d3$functionality))

## Drop 2011 reviews - only 3
d4 <- subset(
  d4,
  year_2011!=1
)
# d4 <- select(d4, -c(year_2011))

d4 <- d4 %>%
  group_by(locationId, year) %>%
  mutate(count = n()) %>%
  mutate(functionality_ratio = sum(functionality)/count)

## Checks
# summary(d4$functionality_ratio)
# anyNA(d4$functionality_ratio)

x1 <- subset(d4, select = c(x_regressors, dl_year, dl_nation))
x1 <- subset(x1, select = -c(nation_China,
                             
                             year_2011,
                             year_2012,
                             year_2013,
                             year_2014))

# x1 <- subset(x1, select = -c(nation_Indonesia))
# x1 <- subset(x1, select = -c(year_2012, year_2013))

# summary(x1)
d4 %>% 
  group_by(year)%>% 
  summarise(
    count=n()
  )

y1 <- d4$functionality_ratio

m1 <- frm(y1, x1, linkfrac = 'logit', var.type =  "cluster", var.cluster =d4$locationId)

sink('frm_m22.txt')
m1r <- frm.pe(m1)
sink()
write.csv(m1r,'frm_m22.csv')

# model11<-lm(d4$functionality_ratio~d4$translated+d4$poi_government+d4$poi_lodging+d4$poi_retail+d4$network_du+d4$nation_HK.Macao+d4$nation_Japan+d4$nation_Malaysia+d4$nation_Philippines+d4$nation_Singapore+d4$nation_South.Korea+d4$nation_Taiwan+d4$nation_Thailand+d4$year_2015+d4$year_2016+d4$year_2017+d4$year_2018+d4$year_2019+d4$year_2020+d4$year_2021)
# summary(model11)


