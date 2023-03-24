# Power Outage Duration Prediction
**by David Sun & Yijun Luo**

>Our Data Analysis on this Power Outage dataset can be found [here](https://jackkkkkkdzk.github.io/Power-Outage-Investigation/).

# Framing the Problem
When power outages happen, what citizens care most about is how long the outage is going to last, and when can they expect the power to be restored. Knowing the rough duration of a power outage according to known variables woudl allow impacted citizens to make better use of the time instead of waiting anxiously. This make the prediction of power outage duration a very meaningful task. We are trying to build a machine learning model that best predicts the duration of power outages. 

At the time of prediction(when outage happens), we are able to know the start time of an outage (MONTH,OUTAGE.START.DATE), where it happens (NERC.REGION, CLIMATE.REGION), the general climate information of that region (CLIMATE.CATEGORY), whether there is a hurricane present at that time (HURRICANE.NAMES). Last but not least, we can have a big picture of what cause the power outage in a short time by simple investigation (CAUSE.CATEGORY).

**Prediction Problem: What will be the severity, measured by the outage duration, of a major power outage?**

**Type: Regression**

**Reponse Variable: OUTAGE.DURATION**

- We choose OUTAGE.DURATION as the response variable. As one of the three only attributes describing the impact of a power outage, (OUTAGE.DURATION, DEMAND.LOSS, CUSTOMERS.AFFECTED), OUTAGE.DURATION has the least proportion of missing values, discovered during our EDA process. A low amount of missing data would be beneficial to our model building, as we wish to use as much data as possible for training. 

**Metric: R<sup>2</sup>**

- For a linear regression model, we can use one of two common metrics to assess the performance of our prediction model, R<sup>2</sup> and RMSE. Both are equally valid metrics, but RMSE is often hard to interpret in relation to the original data. We choose R<sup>2</sup> as the metric to evaluate our model because it is a direct and easy-to-understand measure of how well our prediction fits the response data, ranging between [0,1], with a higher value corresponding to a higher accuracy. 

<u>The following DataFrame is the first ten row of cleaned outage data for the use of ML model.</u>
Most of the cleaning process is from EDA part. The additional cleaning step here is that we remove the outlier of OUTAGE.DURATION by only preserving the first 99% quantile of duration data in our DataFrame

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
Our baseline model employs basic <u>linear regression</u> on features derived from **onehot encoded** categorical data, which are the **'MONTH', 'NERC.REGION', 'CAUSE.CATEGORY'** columns in our cleaned dataset. Although the 'MONTH' column contains all integers, we use it as a ordinal categorical variable, while the 'NERC.REGION' and 'CAUSE.CATEGORY' colummns are nominal categorical values. These three columns showed the most representative information about outage duration during our EDA phase, and the easiest way to implement these variables is to pass them through a sklearn OneHotEncoder() transformer. 

Our baseline model performs with an average R<sup>2</sup> score of 0.28 for our training data and <u>0.23 for our testing data</u>. With only encoding three columns, this result isn't too bad. However, it could be seen that our model's predictions aren't aligning well with the actual durations, with only around one-fourth of the variation explained by our model's regression. This model still needs significant improvement to achieve useful prediction accuracy.




# Final Model
### Improvement

Final Features: MONTH, NERC.REGION, CAUSE.CATEGORY, OUTAGE.START.DATE,CLIMATE.REGION, CLIMATE.CATEGORY,HURRICANE.NAMES

For our final model, one of the first improvements we did, was to include **two additional columns/features** to our OneHotEncoder transformer, 'CLIMATE.REGION' and 'CLIMATE.CATEGORY'. These two columns contain essential information about the climate conditions each power outage had experienced, and could contribute to the severity of outages caused by bad weather. For example, northern regions would most likely suffer from longer outage durations due to cold winter storms than southern regions. 

The next improvements we made was to **add our own custom transformers to address missing details from the original data**. Since most of the columns contained within the cleaned data are categorical data, we would want to include some quantitative data to better assist our regression model. As such, we created a FunctionTransformer from sklearn to map the CAUSE.CATEGORY column to values indicating the mean outage duration by each category. The reason for choosing the CAUSE.CATEGORY column was because it was the most significant and representative feature for our prediction task, and adding a quantitative side to this column would give different severity measures for different causes. 

**Another FunctionTransformer we created is converting the datetime object from the OUTAGE.START.DATE to a binary classification**, of whether the outage happened on a weekend or not. This is a reasonable addition, as we have discovered, outages occuring during the weekend usually takes longer for the mechanics to restore because of different work hours to weekdays. 

**The final FunctionTransformer we created is identifying if the HURRICANE.NAMES column is missing data or not**. If there is a hurricane name corresponding to a particular outage, that usually implies a potentially severe, and long power outage, when comparing to those who aren't caused by hurricanes. In our exploration, we found that the median duration length of outages caused by hurricane is roughly 7 times the median duration length of outages caused by not hurricanes.


### Selection for Best Model and Hyperparameter

We chose three additional regression models, (KNeighborsRegressor, RandomForestRegressor, DecisionTreeRegressor) to compare performance with our original linear regression model. These were chosen as they are popular models to use in classification, so we used the regressor variants to assess the performance of our model. First, we ran a manual iteration test for the max_depth paramter of the DecisionTreeRegressor, and found that a max_depth of 3 performs with the best testing score. We then compared the max_depth 3 DecisionTreeRegressor with the other regressors, and **found out linear regression seems to have the best testing score out of the rest.** Some other regressors perform significantly better on the training data, they don't generalize well to unseen testing data, meaning that they might be too complicated. While our linear regression model has a training accuracy of 0.312, and testing accuracy of 0.257, which is quite close.

In the end, our final linear regression model's testing score has improved by about 0.02 comparing to the baseline model. This improvement isn't much, but it's consistent. This means that to predict the outage duration accurately and reliably, it requires much more data and better feature engineering than our current model.




# Fairness Analysis
#### Does our final model performs better/worse for the west coast or east coast of U.S.?
More specifically, comparing model's prediction performance from states that have CLIMATE.REGION classified as Northeast or West. A fair model should perform equally good on different groups. In our case, it means if our final linear regression model is fair enough, it should result in same or close R<sup>2</sup> score for different regions (states in the West and states in the Northeast). To compare R<sup>2</sup> across two regions, we conduct a permutation test.

**Group X and Group Y:** Northeast and West in CLIMATE.REGION

**Evaluation Metric:** R<sup>2</sup> scores

**Null Hypothesis:** Our model is fair. The R<sup>2</sup> scores generated from our final prediction model for the West Coast states and the Northeast states ***is roughly the same***. 

**Alternative Hypothesis:** Our model is unfair. The R<sup>2</sup> scores generated from our final prediction model for the West Coast states and the Northeast states ***is different***. 

**Test Statistic:** Difference in R<sup>2</sup> scores.

**Significance Level:** 0.05

<iframe src="assets/permu_test.html" width=800 height=600 frameBorder=0></iframe>

**Resulting P-value:** 0

**Conclusion:** Since the P-value is below the significance level, we reject the Null Hypothesis "Our model is fair. The R<sup>2</sup> scores generated from our final prediction model for the West Coast states and the Northeast states ***is roughly the same***". It seems like our final model is not fair enough yet. There is still space for improvement.