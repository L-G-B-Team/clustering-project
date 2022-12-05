# Goal of This Project: 

## We are continuing working on using regression modeling to make predictions based on our Zillow data of 2017 properties for single family homes. In this itteration of the project we are using clustering algorithms to aid us in determining similarites of the data. We are also shifting our focus to logerror rather than tax value.

### Project Plan:

 #### Acquire and Prepare Data
   1. Data is brought in from either the SQL database using a locally stored env file that must contain the username, password, and host name in a get_bd_url function OR via a locally strored csv file.
   2. Data is prepared in order for our team to explore and modified to use in modeling
   3. Data is split into three dataframes called train, validate, and test so that we can train and explore on in-sample data and use out-of-sample to validate and finally test. 

 #### Exploration of Date
   1. Use visualizations to see patterns in the data and try to determine drivers/indicators of logerror
   2. Use statistical testing, if necessary, to confirm or reject the relationship between features
   3. Use clustering to create new categorical features that can be explored 
   4. Determine what features align with logerror deviating significantly from zero
   
 #### Modeling
   1. Take features determined to be significant in exploration
   2. Determine which regression algorithm to use
   3. Run 4 models with chosen features and determine the best one by evaluating the RMSE score on the train set and run on the validate set
   4. The best model will be deterined by lowest RMSE on train, and lowest variance between the RMSE of train and validate
   5. Run the top single model on the test dataset
   6. Evaluate results
   
 #### Recommendations / What Comes Next
   1. Based on model evaluation of the test dataset make recommendations and predictions
   2. Recommendations will include:
       * adding a member to our group to help us focus
       * beginning with a fully developed plan of action
       * experimenting with dropping columns and rows with more or less null values, and trying different ways to impute for nulls
   3. Further steps are
       * experimenting more with different clustering features
       * taking more time to analyze features that show strong individual correlation with logerror and seeing how they fit into clusters
   
 #### Wrangling the Data (from SQL database using env file or local csv)
For this we chose to initailly eliminate columns with more than 50% of the values being Null, and rows with anymore than 25% Null values. We also removed 'property county land use code' and 'property zoning desc' as they could not be converted from objects. Any further nulls were imputed with zero. We also made the data more human readable by changing the names of the columns.


# Data Dictionary
| Feature | Definition | Manipulation |Data Type|
|:--------|:-----------|:-----------|:-----------|
|<img width=50/>|<img width=100/>|<img width=50/>|<img width=150/>|
|||**Numerical Data**
|<img width=50/>|<img width=100/>|<img width=50/>|<img width=150/>|
|*calc_bath_and_bed*|  The number of Bathrooms | Changed the name| float
|*calc_sqft*|  The square footage of the property | Name change| float
|*fips*| The code of geographic location | data type conversion| categorical
|*fireplace_count*| The number of fireplaces on a property | Null values imputed as zero| float
|*garage_car_count*| The number of cars a gargae can hold | Null values imputed as zero| float
|*lot_sqft*| The square footage assigned to a property's exterior | Null values imputed as zero| float
|*pool_count*| Number of pools on a property | Null values imputed as zero| float
|*tax_value*| The dollar amount paid in taxes for the property |Null values imputed as zero | float
|**Target Data**
||<img width=150/>|<img width=550/>|
|**log_error** | **The calculated logarithmic difference between estimated sale price and actual sale price** || **float**



## Our Thought Process:
   1. We began by taking a look at our train dataset in relation to our target vaiable, logerror.
   2. We then look to see what individial features tend to deviate significantly from zero logerror. 
   3. From looking at our features, we were able to generate four specific questions to ask of the data. 
   4. We used our human eyes and computer algorithms to create clusters which were tested against the target variable.
   5. We used visualizations and statistical tests to answer our questions. Sometimes the evaluation and answers lead to further questions. 
   
## MODELING
Moving forward in our modeling we chose to use tax value and calculated square feet as a clustered feature, along with the features fireplace count, latitude, and longitude. We chose to include these in our model because all six features showed to consistently coincide with having clusters that were both centered around zero logerror as well as having wide variance away from zero logerror. We chose to use tax value and calculated square feet as a cluster because together they showed the most centralized cluster around zero with the fewest significant outliers.

 ### Modeling Pre-Processing
   1. First we decided what features would move into our model.
      * we chose to use tax value and calculated square feet as a clustered feature, along with the features fireplace count, latitude, and longitude.
      * We chose to use tax value and calculated square feet as a cluster because together they showed the most centralized cluster around zero with the fewest significant outliers.
   2. We ensured all of our data was numerical
   3. We split our train, validate, and test data sets to seperate the target variable of logerror to use as y- train, validate, and test.
  
  ## Our model run on the test data results in RMSE score that is 12% lower than baseline
  
  ## Based on our analysis and modeling:
* Our model performed better than baseline on the data, but we are still not happy with it
* We feel that the amount of nulls in the original data leads to skewed results due to being able to use limited features
* Future itterations of this project should include a more focused plan, and more time to explore relationships between features
* We conclude that clustering can be a useful tool in finding unseen patterns in the data, but more time would be appreciated to fully dive into how this method could improve predictions






