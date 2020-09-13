# Capstone-Project: Examining Bias in Chicago Policing with Machine Learning
============================================================================

## Project Scope
The intention of this project is to explore policing data for bias, measured in arrest rates. Iterating through various machine learning models, I attempted to answer two questions:
1. Knowing zip code, time of day and race of the person investigated, can we predict whether or not an incident resulted in arrest?
2. Knowing which officers from our dataset had the highest arrest rates, can we predict whether or not one of the "highest" officers was involved, using only demographic information about the person investigated?

There is a wealth of policing data recently available online, and I used the city of Chicago's Investigative Stop Reporting. This is a detailed, large (290k+ row) dataset covering January 2018 through December 2019. Rare among publically available police datasets, it includes randomized IDs for first, second and supervisory officers involved in each investigative stop. 

Given the nature of self-reported data, there is good reason to be skeptical about anything in this dataset that could be subjected to bias, from any perspective.
For this reason, I've chosen to focus on arrests as an objective outcome. Unlike (for example) perceived threats to the officer or violent encounters, this measure should represent the outcome of the situation, whether or not all events in between are objective.

# Overview

This project follows [OSEMN process](https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492), which is an acronym for Obtain Scrub Examine Model Interpret. The steps are below.

  __1. Data Collection__
  
   Data sources: [Chicago Investigative Stop Report data](https://home.chicagopolice.org/statistics-data/isr-data/) and Demographic information from the [US Census website](https://data.census.gov/cedsci/map?q=60624&tid=ACSDP5Y2018.DP05&vintage=2018&layer=VT_2018_860_00_PY_D1&cid=DP05_0001E&mode=thematic).
    
   The ISR data includes 290k+ rows of police-reported stop data, and 171 columns of input. Demographic details are included (age, sex, race) as well as geographical details (business ward, intersection), details on the stop (was a search conducted? if so, did the officer receive consent to search) and various potenital "outcome" fields (result of stop was arrest, inventory number and details of confiscated property)
    
   These columns are a mixture of binary Y/N (is a violent crime suspected?), categorical (CPD Beat) and even freeform text (zip code, vehicle model description). 
    
  __2. Cleaning and EDA__
  
  This dataset included a large number of missing values. Columns not impacting analysis were dropped, including all columns referring to the CPD internal inventory number of confiscated items. Some columns were entirely empty or only had a handful of values; those were dropped as well. 
  
  For the scope of this analysis, columns that included only text descriptions were also removed. The "clean" .csv file saved to this repository represents these changes. 
  
  This initial cleaning allowed for an early look at enforcement outcome vs. contact hour: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2011.37.59%20AM.png)
  
  As well as enforcement outcome by reported race: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2011.38.14%20AM.png)
  
  Additional cleaning measures will be covered below, with additional details on the machine learning methods chosen.
  
  
  __3. Machine Learning Model #1__ 
  
  __Knowing zip code, time of day and race of the person investigated, can we predict whether or not an incident resulted in arrest?__
  
  ### Additional Cleaning Methods
  
  
  The intention of this model was to start with just a few columns of data; based on exploratory visualizations a new dataframe was limited to contact hour, race code and zipcode. 
  
  Rows missing zipcode or enforcement type were removed for this reason. This left 67k+ rows for our model to run through. The target for this model was whether or not an arrest was reported (aka whether ENFORCEMENT_TYPE = ARR). Both zipcode and race had separate label encoders created, so the end results could be interpreted. 
  
  ### Model Details
  
  Logistic Regression, Gradient Boosting, XGBoost and Decision Tree, AdaBoost and Support Vector Classification models were run, and all had F1 scores between .69 and .77. The highest performing models were Gradient Boosting and XGBoost.
  
  The feature importance of the XGBoost model can be seen below: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2012.08.51%20PM.png)
  
  Using our zipcode label encoder's `inverse_transformation`, we now know that the highest contributing factors are a stop occuring in 60624 or 60623.
  
  
  ### Statistical Significance
  
  ### Interpretation
  
  Using the Census website gives us additional context into why 60624 and 60623 may indicate higher rates of arrest - these are overwhelmingly communities of color with large numbers of residents living below the poverty line. Doing a quick calculation on these zipcodes show that the arrest rate is almost 2x the arrest rate for the entire dataset.
  
  We can clearly see these zip's outsize impact on the dataset viewed below: ![image](https://github.com/lambertmk/Capstone-Project/blob/master/images/Screen%20Shot%202020-09-13%20at%2012.16.25%20PM.png)
  
  
  


  __4. Machine Learning Model # 2__ - can we predict whether or not an incident included one of the 100 officers with the highest arrest percentage?
  __5. Interpretation - or, what do we do with the model's findings?__
