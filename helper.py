# This application calculates Pearson coefficient for the correlation
import math
from array import array

def mean(ar):
    return sum(ar) / len(ar)

x=[1,2,3,4,5]
y=[1,2,3,4,5]

meanx = mean(x)
meany = mean(y)

def covariance(ar1, ar2):
   return sum( (ar1[i]-meanx) * (ar2[i]-meany) for i in range(len(ar1)))

def sigma(ar):
    return math.sqrt(sum( (ar[i] - meanx) ** 2 for i in range(len(ar))))

def correlation(ar1, ar2):
    return covariance(ar1, ar2) / (sigma(ar1) * sigma(ar2))

print(correlation(x,y))