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