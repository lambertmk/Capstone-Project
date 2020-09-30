# Capstone-Project: Examining Bias in Chicago Policing with Machine Learning
==================================================================================

## Project Scope
The intention of this project is to explore policing data for bias, measured in arrest rates. Iterating through various machine learning models, I attempted to answer two questions:
1. Knowing zip code, time of day and race of the person investigated, can we predict whether or not an incident resulted in arrest?
2. Knowing which officers from our dataset had the highest arrest rates, can we predict whether or not one of the "highest" officers was involved, using only demographic information about the person investigated?

There is a wealth of policing data recently available online, and I used the city of Chicago's Investigative Stop Reporting. This is a detailed, large (290k+ row) dataset covering January 2018 through December 2019. Rare among publicly available police datasets, it includes randomized IDs for first, second and supervisory officers involved in each investigative stop. 

Given the nature of self-reported data, there is good reason to be skeptical about anything in this dataset that could be subjected to bias, from any perspective.
For this reason, I've chosen to focus on arrests as an objective outcome. Unlike (for example) perceived threats to the officer or violent encounters, this measure should represent the outcome of the situation, whether or not all events in between are objective.

# Overview

This project follows [OSEMN process](https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492), which is an acronym for Obtain Scrub Examine Model Interpret. The steps are below.

  __1. Data Collection__
  
   Data sources: [Chicago Investigative Stop Report data](https://home.chicagopolice.org/statistics-data/isr-data/) and Demographic information from the [US Census website](https://data.census.gov/cedsci/map?q=60624&tid=ACSDP5Y2018.DP05&vintage=2018&layer=VT_2018_860_00_PY_D1&cid=DP05_0001E&mode=thematic).
    
   The ISR data includes 290k+ rows of police-reported stop data, and 171 columns of input. Demographic details are included (age, sex, race) as well as geographical details (business ward, intersection), details on the stop (was a search conducted? if so, did the officer receive consent to search) and various potential "outcome" fields (result of stop was arrest, inventory number and details of confiscated property)
    
   These columns are a mixture of binary Y/N (is a violent crime suspected?), categorical (CPD Beat) and even freeform text (zip code, vehicle model description). 
    
  __2. Cleaning and EDA__
  
  This dataset included a large number of missing values. Columns not impacting analysis were dropped, including all columns referring to the CPD internal inventory number of confiscated items. Some columns were entirely empty or only had a handful of values; those were dropped as well. 
  
  For the scope of this analysis, columns that included only text descriptions were also removed. The "clean" .csv file saved to this repository represents these changes. 
  
  This initial cleaning allowed for an early look at enforcement outcome vs. contact hour: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2011.37.59%20AM.png)
  
  As well as enforcement outcome by reported race: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2011.38.14%20AM.png)
  
  Additional cleaning measures will be covered below, with additional details on the machine learning methods chosen.
  
  
 ## __3. Machine Learning Model #1__ 
  
  __Knowing zip code, time of day and race of the person investigated, can we predict whether or not an incident resulted in arrest?__
  
  ### Additional Cleaning Methods
  
  
  The intention of this model was to start with just a few columns of data; based on exploratory visualizations a new dataframe was limited to contact hour, race code and zipcode. 
  
  Rows missing zipcode or enforcement type were removed for this reason. This left 67k+ rows for our model to run through. The target for this model was whether or not an arrest was reported (aka whether ENFORCEMENT_TYPE = ARR). Both zipcode and race had separate label encoders created, so the end results could be interpreted. 
  
  ### Model Details
  
  Logistic Regression, Gradient Boosting, XGBoost and Decision Tree and AdaBoost models were run, and all had F1 scores between .69 and .77. The highest performing models were Gradient Boosting and XGBoost.
  
  The feature importance of the XGBoost model can be seen below: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2012.08.51%20PM.png)
  
  Using our zipcode label encoder's `inverse_transformation`, we now know that the highest contributing factors are a stop occuring in 60624 or 60623.
  
  
  ### Interpretation
  
  Using the Census website gives us additional context into why 60624 and 60623 may indicate higher rates of arrest - these are overwhelmingly communities of color with large numbers of residents living below the poverty line. Doing a quick calculation on these zipcodes show that the arrest rate is almost 2x the arrest rate for the entire dataset.
  
  We can clearly see these zip's outsize impact on the dataset viewed below: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2012.16.25%20PM.png)
  
  Our visualization clearly display what we could have logically expected - of the four possible outcomes, the majority of stops result in arrest. Of stops that result in arrest, the race of the person is overwhelmingly Black. ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-15%20at%202.48.13%20PM.png)
  
  ### Statistical Significance
  
  It's helpful to know that these two zips have high rates of arrest; next, let's examine if those differences are statistically significant. Additionally, `MultiComparison` and tukey makes it easy to iterate through our possible combinations and determine which have a p value of less than .05. A snippet is below: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2012.23.26%20PM.png)
  
  Using Ordinary Least Squares, we've determined that there is a statistically significant difference in arrest rates by zip code, as well as by race. Using tukey, we're able to iterate through a list of each and return a true or false value for each pairing.
  
  

## __4. Machine Learning Model # 2__ 
  
__Knowing which officers from our dataset had the highest arrest rates, can we predict whether or not one of the "highest" officers was involved, using only demographic information about the person investigated?__

### Additional Cleaning Methods

This model had a different focus than the first one, and used the data in a very different way. Instead of starting with arrest as our target and looking at only three variables, I've included *all* demographic information possible. This required a good deal more cleaning and manipulation.

The first step was to isolate the arrest rate for all officers, so we could determine the officers with the highest rate. Here's what that looks like: ![image](https://github.com/lambertmk/Using-Machine-Learning-to-Examine-Bias-in-Chicago-Policing-Data/blob/master/images/Employee%20ID%20vs.%20Arrest%20Rate.png)

Many assumptions were made here, so that the resulting dataset was able to be analyzed using common machine learning algorithms. Freeform text fields were largely ignored, as were fields dependent on the discovery of a particular item or substance (these were quite small in number). Columns such as race, build and hair color were left as categorical, and a label encoder applied.

The cleaned dataframe result was roughly 48 columns; after dummy columns were created, it was 394. Here's a view of the correlation, before one hot encoding: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%202.02.55%20PM.png)

  ### Model Details
  
  Logistic Regression, Gradient Boosting, XGBoost, Decision Tree, AdaBoost and Random Forest Classifier were run; the best result was achieved using Decision Tree, with an F1 score of roughly .33. A grid search was performed to boost DT's performance, to no luck. 
  
  This dataset is extremely imbalanced; the highest 100 officers make up only 1.2% of the dataset! For this reason, Decision Tree was also run using `class_weight = 'balanced'`; the result was not any better than the original run. 
  
  A visual representation of this Decision Tree model is included in the image folder. 
  
  ### Interpretation
  
Examining arrest rates compared to the entire dataset of reported race leads to a stark discovery - people reported as black are arrested at 8% (raw) higher than their representation in the dataset. For white people, it's 6% lower! Every single other race besides black has a lower percentage of arrests than total representation in the data.

Much of this analysis is centered around the concept of a "highest" officer; that is, one of the 100 officers with the highest arrest rate. The following shows those officers' relationship to race. The data is limited to arrests only. ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-15%20at%202.57.36%20PM.png)

The outcome of the Decision Tree model is an F1 score of .34 - not a strong model. Given that we're trying to predict whether or not one of 100 officers was involved and the dataset includes 7936 officers total (1.2%), we could also say that this model performs roughly 27x better than random guessing. 

The intention of this second model was to test for strong bias in policing; to look for patterns in demographic data that would predict whether or not a subset of officers was involved. Knowing that our best result is a 34% F1 score is actually resassuring - we cannot in good conscience pinpoint if bias is occuring with these officers.

 ### Guide to Enforcement Types
In the Chicago PD, there are four types of enforcement outcomes: ANOV, ARR, OTH and PSC. Of course, this could also be missing from a particular row in the  dataset. 
 
 ANOV: Administrative Notice of Violation - a citation/ticket
 
 ARR: Arrest
 
 OTH: Other
 
 PSC: This is undefined in the Data Dictionary and on Chicago PD's website, but is likely a citation
 
 Definitions taken from the City of Chicago's [Glossary of Terms](https://www.chicago.gov/city/en/depts/ah/supp_info/glossary_of_terms.html)

 ### Reproduction
 
As mentioned above, the city of Chicago provided the data for this analysis: [Chicago Investigative Stop Report data](https://home.chicagopolice.org/statistics-data/isr-data/) and Demographic information from the [US Census website](https://data.census.gov/cedsci/map?q=60624&tid=ACSDP5Y2018.DP05&vintage=2018&layer=VT_2018_860_00_PY_D1&cid=DP05_0001E&mode=thematic).

To reproduce any of this analysis, I've saved this data in the folder titled Data Files. The cleaned data (with freeform text removed) can be found as ["ISR-data_clean.csv.zip"](https://github.com/lambertmk/Using-Machine-Learning-to-Examine-Bias-in-Chicago-Policing-Data/blob/master/Data%20Files/ISR_data_clean.csv.zip)

The data used for the first set of Machine Learning models is saved as ["ISR_data_model1.csv"](https://github.com/lambertmk/Using-Machine-Learning-to-Examine-Bias-in-Chicago-Policing-Data/blob/master/Data%20Files/ISR_data_model1.csv). If starting here, proceed with train_test_split.

The data used for the second set of models can be found in the same Data Files folder, with the title ["ISR_data_model_2.csv.zip"](https://github.com/lambertmk/Using-Machine-Learning-to-Examine-Bias-in-Chicago-Policing-Data/blob/master/Data%20Files/ISR_data_model2.csv.zip). Note that this is a good deal larger!

Similar datasets can be found via the Police Data Accountability Project. [PDAP Github here.](https://github.com/Police-Data-Accessibility-Project/Police-Data-Accessibility-Project)

## __Next Steps & Additional Work__ 

Given the size of this dataset, there are a great deal of questions that could be asked and answered. What's covered above is a small slice. Some potential further work:
1. Use time series analysis and ARIMA (or similar) to predict the number of arrests by zip code, officer, etc.
2. Use time series analysis to discover seasonal, monthly variability in stop rates
3. Analyze freeform columns (eg notes on found items) to predict arrest
4. Utilize resampling to optimize model performance above
5. Attempt to reduce computing power needed with PCA for first model
