# Power Outage Duration Prediction
**by David Sun & Yijun Luo**


# Framing the Problem
When power outage happens, what citizens care most about is how long the outage is going to last, when will their life be back to normal. When they can expect the duration of outage, they can make best use of the time without power instead of waiting anxiously. This make the prediction of power outage duration a very meaningful task. We are trying to build a machine learning model that best predict the duration of power outage. 

At the time of prediction(when outage happens), we are able to know the start time of an outage (MONTH,OUTAGE.START.DATE), where it happens (NERC.REGION, CLIMATE.REGION), the general climate information of that region (CLIMATE.CATEGORY), whether there is a hurricane present at that time (HURRICANE.NAMES). Last but not least, we can have a big picture of what cause the power outage in a short time by simple investigation (CAUSE.CATEGORY) 

- **Prediction Problem: How long will a power outage last?**
- **Type: Regression**
- **Reponse Variable: OUTAGE.DURATION**
    - The reason we choose OUTAGE.DURATION as the response variable it corresponds to out prediction problem.
- **Metric: R<sup>2</sup>**
    - The reason we choose R<sup>2</sup> as the metric to evaluate our model is that it is one useful and important way to measure how our prediction is explained by the features we use in a linear regression model





# Baseline Model
### Linear Regression Model 1.0


### Features
**CAUSE.CATEGORY, MONTH, NERC.REGION, CLIMATE.REGION, CLIMATE.CATEGORY**
including how many are quantitative, ordinal, and nominal, 

### Encoding and Transformation

### Performance
Report the performance of your model and whether or not you believe your current model is “good” and why.




# Final Model
### Features 
**MONTH, NERC.REGION, CAUSE.CATEGORY, OUTAGE.START.DATE,CLIMATE.REGION, CLIMATE.CATEGORY,HURRICANE.NAMES**





# Fairness Analysis
