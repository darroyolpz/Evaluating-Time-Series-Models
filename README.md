# Evaluating Time Series Models

Outline that explains the purpose, features, functionality, and behavior of a product. It guides business and technical teams through the process of building, launching, and marketing a product based on customer needs.

## MAE (Mean Absolute Error)

$MAE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |y_i-\hat{y}_i|$

MAE, also known as L1 Loss, can be calculated by dividing the sum of the absolute differences between the predicted values and the actual values by the number of samples(n). Since this is the process for calculating an average, from now on we will refer to this as 'calculating the mean'. Since the scale of MAE is the same scale as the target variable being predicted, the meaning of the value can be understood intuitively.

```python
import numpy as np #import numpy package

def MAE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MAE(TRUE, PRED)
```

## MSE (Mean Squared Error)

$MSE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2$

$RMSE=\sqrt{\frac{1}{n}\displaystyle\sum_{i=1}^{n} (y_i-\hat{y}_i)^2}$

MSE, also known as L2 Loss, is calculated by taking the mean of the squared differences between the predicted values and the actual values. The more the predicted values deviate from the actual values, the more the MSE value will increase. It will increase exponentially. Since the calculated value is squared, the scale of the target variable and MSE is different. In order to match the scale of the target value, we need to calculate the square root of the MSE. This value is called RMSE.

```python
def MSE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.square(true-pred))

TRUE = np.array([10, 20, 30, 40, 50])
PRED = np.array([30, 40, 50, 60, 70])

MSE(TRUE, PRED)

RMSE = np.sqrt(MSE(TRUE, PRED))
```

## MAPE (Mean Absolute Percentage Error)

$MAPE=\frac{1}{n}\displaystyle\sum_{i=1}^{n} |\frac{y_i-\hat{y}_i}{y_i}|$

In order to calculate MAPE, first calculate the relative size of the error compared to the actual values by dividing the difference between each of the actual values and the predicted value by each actual value. Then, take the absolute value of the relative size of the error for each actual value and calculate the mean. Since the size of the error is expressed as a percentage value, it can be used to understand the performance of the model. Also, it is a suitable metric for evaluating a model when there is more than one target variable because the scale of the calculated errors across the target variables will be similar.

However, if there is an actual value of 0, MAPE will be undefined. In addition, even if the absolute values of the errors are same, more penalties are added to a predicted value that overestimates.

```python
def MAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean(np.abs((true-pred)/true))

TRUE_UNDER = np.array([10, 20, 30, 40, 50])
PRED_OVER = np.array([30, 40, 50, 60, 70])
TRUE_OVER = np.array([30, 40, 50, 60, 70])
PRED_UNDER = np.array([10, 20, 30, 40, 50])


print('Comparison between MAE, MAPE when average error is 20 depending on the relationship between actual and predicted value \n')

print('When actual value is smaller than predicted value (Overestimating)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('MAPE:', MAPE(TRUE_UNDER, PRED_OVER))


print('\nWhen actual value is bigger than predicted value (Underestimating)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('MAPE:', MAPE(TRUE_OVER, PRED_UNDER))
```

MAPE divides the error by the actual value $y$ to convert it to a percentage. Therefore, the calculated value is dependent on $y$. Even if the numerators are the same, smaller denominators will increase the overall error.

We can observe this phenomenon by observing the two examples above where (`TRUE_UNDER`, `PRED_OVER`) predicts values that are more than the actual values by 20 and (`TRUE_OVER`, `PRED_UNDER`) predicts values that are less than the actual values by 20. On both examples, the MAE values are the same at 20. However, for the `TRUE_UNDER` case the MAPE value is calculated as 0.913 and `TRUE_OVER` case calculates the MAPE value as 0.437.

## SMAPE (Symmetric Mean Absolute Percentage Error)

$SMAPE=\frac{100}{n}\displaystyle\sum_{i=1}^{n} \frac{|y_i-\hat{y}_i|}{|y_i| + |\hat{y}_i|}$

```python
def SMAPE(true, pred):
    '''
    true: np.array 
    pred: np.array
    '''
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred))) #we won't include 100 in this code since it's a constant

print('Comparison between MAE, SMAPE when average error is 20 \n')

print('When actual value is smaller than predicted value (Overestimating)')
print('MAE:', MAE(TRUE_UNDER, PRED_OVER))
print('SMAPE:', SMAPE(TRUE_UNDER, PRED_OVER))


print('\nWhen actual value is bigger than predicted value (Underestimating)')
print('MAE:', MAE(TRUE_OVER, PRED_UNDER))
print('SMAPE:', SMAPE(TRUE_OVER, PRED_UNDER))
```

We can observe that MAPE produced different values of 0.91 and 0.43 respectively on the same example, but SMAPE yielded the same values of 0.29. However, SMAPE is dependent on $\hat{y}_i$ because the predicted value $\hat{y}_i$ is included in the denominator. When the predicted value is an underestimation, the denominator becomes smaller and the overall error increases.


## RMSEE (Root Mean Squared Scaled Error)

$RMSSE=\sqrt{\displaystyle\frac{\frac{1}{h}\sum_{i=n+1}^{n+h} (y_i-\hat{y}*i)^2}{\frac{1}{n-1}\sum*{i=2}^{n} (y_i-y_{i-1})^2}}$

$y_i$: Actual value to be predicted
$\hat{y}_i$: Value predicted by the model
$n$: Size of the training dataset
$h$: Size of the test dataset

RMSSE is a modified form of Mean Absolute Scaled Error and solves the problems of MAPE and SMAPE mentioned above. We have seen from above examples that MAPE and SMAPE result in an uneven overall error depending on the underestimation or overestimation of the model since they use the actual and predicted values of the test data to scale the MAE.

```python
def RMSSE(true, pred, train): 
    '''
    true: np.array 
    pred: np.array
    train: np.array
    '''
    
    n = len(train)

    numerator = np.mean(np.sum(np.square(true - pred)))
    
    denominator = 1/(n-1)*np.sum(np.square((train[1:] - train[:-1])))
    
    msse = numerator/denominator
    
    return msse ** 0.5

TRAIN = np.array([10, 20, 30, 40, 50]) #create a random training dataset for calculating RMSSE

print(RMSSE(TRUE_UNDER, PRED_OVER, TRAIN))
print(RMSSE(TRUE_OVER, PRED_UNDER, TRAIN))
print(RMSSE(TRUE2, PRED2_OVER, TRAIN))
print(RMSSE(TRUE2, PRED2_UNDER, TRAIN))
```