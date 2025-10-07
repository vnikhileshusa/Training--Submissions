import numpy as np
from sklearn.linear_model import LinearRegression
# House size (sqft)
X = np.array([[1000], [1500], [2000], [2500], [3000]])

# House price (in $1000s)
y = np.array([200, 250, 300, 350, 400])
# Create the model
model = LinearRegression()

# Train the model using our data
model.fit(X, y)
predicted_price = model.predict([[2200]])
print("Predicted house price for 2200 sqft: $", predicted_price[0]*1000)
