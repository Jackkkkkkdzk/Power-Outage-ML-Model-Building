# Power Outage Duration Prediction
**by David Sun & Yijun Luo**


# Framing the Problem
When power outages happen, what citizens care most about is how long the outage is going to last, and when can they expect the power to be restored. Knowing the rough duration of a power outage according to known variables woudl allow impacted citizens to make better use of the time instead of waiting anxiously. This make the prediction of power outage duration a very meaningful task. We are trying to build a machine learning model that best predicts the duration of power outages. 

At the time of prediction(when outage happens), we are able to know the start time of an outage (MONTH,OUTAGE.START.DATE), where it happens (NERC.REGION, CLIMATE.REGION), the general climate information of that region (CLIMATE.CATEGORY), whether there is a hurricane present at that time (HURRICANE.NAMES). Last but not least, we can have a big picture of what cause the power outage in a short time by simple investigation (CAUSE.CATEGORY).

- **Prediction Problem: What will be the severity, measured by the outage duration, of a major power outage?**
- **Type: Regression**
- **Reponse Variable: OUTAGE.DURATION**
    - We choose OUTAGE.DURATION as the response variable. As one of the three only attributes describing the impact of a power outage, (OUTAGE.DURATION, DEMAND.LOSS, CUSTOMERS.AFFECTED), OUTAGE.DURATION has the least proportion of missing values, discovered during our EDA process. A low amount of missing data would be beneficial to our model building, as we wish to use as much data as possible for training. 
- **Metric: R<sup>2</sup>**
    - For a linear regression model, we can use one of two common metrics to assess the performance of our prediction model, R<sup>2</sup> and RMSE. Both are equally valid metrics, but RMSE is often hard to interpret in relation to the original data. We choose R<sup>2</sup> as the metric to evaluate our model because it is a direct and easy-to-understand measure of how well our prediction fits the response data, ranging between [0,1], with a higher value corresponding to a higher accuracy. 

The following DataFrame is the first ten row of cleaned outage data for the use of ML model

|   YEAR |   MONTH | U.S._STATE   | NERC.REGION   | CLIMATE.REGION     |   ANOMALY.LEVEL | CLIMATE.CATEGORY   | OUTAGE.START.DATE   | OUTAGE.START.TIME   | CAUSE.CATEGORY     | CAUSE.CATEGORY.DETAIL   |   HURRICANE.NAMES |   OUTAGE.DURATION |   DEMAND.LOSS.MW |   CUSTOMERS.AFFECTED |
|-------:|--------:|:-------------|:--------------|:-------------------|----------------:|:-------------------|:--------------------|:--------------------|:-------------------|:------------------------|------------------:|------------------:|-----------------:|---------------------:|
|   2011 |       7 | Minnesota    | MRO           | East North Central |            -0.3 | normal             | 2011-07-01 00:00:00 | 17:00:00            | severe weather     | nan                     |               nan |              3060 |              nan |                70000 |
|   2014 |       5 | Minnesota    | MRO           | East North Central |            -0.1 | normal             | 2014-05-11 00:00:00 | 18:38:00            | intentional attack | vandalism               |               nan |                 1 |              nan |                  nan |
|   2010 |      10 | Minnesota    | MRO           | East North Central |            -1.5 | cold               | 2010-10-26 00:00:00 | 20:00:00            | severe weather     | heavy wind              |               nan |              3000 |              nan |                70000 |
|   2012 |       6 | Minnesota    | MRO           | East North Central |            -0.1 | normal             | 2012-06-19 00:00:00 | 04:30:00            | severe weather     | thunderstorm            |               nan |              2550 |              nan |                68200 |
|   2015 |       7 | Minnesota    | MRO           | East North Central |             1.2 | warm               | 2015-07-18 00:00:00 | 02:00:00            | severe weather     | nan                     |               nan |              1740 |              250 |               250000 |
|   2010 |      11 | Minnesota    | MRO           | East North Central |            -1.4 | cold               | 2010-11-13 00:00:00 | 15:00:00            | severe weather     | winter storm            |               nan |              1860 |              nan |                60000 |
|   2010 |       7 | Minnesota    | MRO           | East North Central |            -0.9 | cold               | 2010-07-17 00:00:00 | 20:30:00            | severe weather     | tornadoes               |               nan |              2970 |              nan |                63000 |
|   2005 |       6 | Minnesota    | MRO           | East North Central |             0.2 | normal             | 2005-06-08 00:00:00 | 04:00:00            | severe weather     | thunderstorm            |               nan |              3960 |               75 |               300000 |
|   2015 |       3 | Minnesota    | MRO           | East North Central |             0.6 | warm               | 2015-03-16 00:00:00 | 07:31:00            | intentional attack | sabotage                |               nan |               155 |               20 |                 5941 |
|   2013 |       6 | Minnesota    | MRO           | East North Central |            -0.2 | normal             | 2013-06-21 00:00:00 | 17:39:00            | severe weather     | hailstorm               |               nan |              3621 |              nan |               400000 |


# Baseline Model
### Linear Regression Model 1.0
We used a scikit-learn LinearRegression() model with features including **CAUSE.CATEGORY, MONTH, NERC.REGION**
including how many are quantitative, ordinal, and nominal, 

### Encoding and Transformation

### Performance
Report the performance of your model and whether or not you believe your current model is “good” and why.




# Final Model
### Features 
**MONTH, NERC.REGION, CAUSE.CATEGORY, OUTAGE.START.DATE,CLIMATE.REGION, CLIMATE.CATEGORY,HURRICANE.NAMES**





# Fairness Analysis
