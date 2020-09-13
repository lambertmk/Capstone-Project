# Capstone-Project


## Project Scope
The intention of this project is to explore policing data for bias, measured in arrest rates. Iterating through various machine learning models, I attempted to answer two questions:
1. Knowing zip code, time of day and race of the person investigated, can we predict of they were arrested?
2. Knowing which officers from our dataset had the highest arrest rates, can we predict whether or not one of the "highest" officers was involved, using only demographic information about the person investigated?

There is a wealth of policing data recently available online, and I used the city of Chicago's Investigative Stop Reporting. This is a detailed, large (290k+ row) dataset covering January 2018 through December 2019. Rare among publically available police datasets, it includes randomized ids for first, second and supervisory officers involved in each investigative stop recorded. 

Given the nature of self-reported data, there is good reason to be skeptical about anything in this dataset that could be subjected to bias, from any perspective.
For this reason, I've chosen to focus on arrests as an objective outcome. Unlike (for example) perceived threats to the officer or violent encounters, this measure should represent the outcome of the situation, whether or not all events in between are objective.

Overview
  1. Data Collection
  2. Cleaning and EDA
  3. Machine Learning Model #1 - using only zip code, contact hour and reported race, can we predict whether or not a incident will result in arrest?
  3a. Are the two highest features statistically significant from the rest of the dataset?
  4. Machine Learning Model # 2 - can we predict whether or not an incident included one of the 100 officers with the highest arrest percentage?
  5. Interpretation - or, what do we do with the model's findings?
  
  This process follows OSEMN process, which is an acronym for Obtain Scrub Examine Model Interpret. More information can be found below.
  https://towardsdatascience.com/5-steps-of-a-data-science-project-lifecycle-26c50372b492
