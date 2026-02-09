# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date:09/02/2026
# 212224040319
# Tesla Stock Price Dataset


### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program

### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Tesla Dataset.csv')

# Convert 'Date' column to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Resample the 'Open' column by year and calculate the mean
# Using 'YE' as per FutureWarning
resampled_data = data['Open'].resample('YE').mean().to_frame()
resampled_data.index = resampled_data.index.year # Extract year from the DatetimeIndex
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year', 'Open': 'OpenPrice'}, inplace=True)

years = resampled_data['Year'].tolist()
prices = resampled_data['OpenPrice'].tolist()



```

### A - LINEAR TREND ESTIMATION


```

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, prices)]
n = len(years)
b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

```

### B- POLYNOMIAL TREND ESTIMATION

```
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

#Visualising results
print(f"Linear trend: y={a:.2f} + {b_poly:.2f}x + {c_poly:.2f}x^2")
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year', inplace=True)

plt.figure(figsize=(8,5))
plt.plot(resampled_data.index, resampled_data['OpenPrice'], color='blue', marker='o', label='Open Price')
plt.plot(resampled_data.index, resampled_data['Linear Trend'], color='black', linestyle='--', label='Linear Trend')
plt.title("Tesla Stock Price - Linear Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Open Price")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(resampled_data.index, resampled_data['OpenPrice'], color='blue', marker='o', label='Open Price')
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], color='red', marker='o', label='Polynomial Trend')
plt.title("Tesla Stock Price - Polynomial Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Open Price")
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT

A - LINEAR TREND ESTIMATION


<img width="1299" height="642" alt="image" src="https://github.com/user-attachments/assets/d5e1c995-4f98-4f02-9b5d-375f4586d64b" />




B- POLYNOMIAL TREND ESTIMATION


<img width="1162" height="589" alt="image" src="https://github.com/user-attachments/assets/e75a2924-aeab-4d0d-a401-6789bb88a6c6" />



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
