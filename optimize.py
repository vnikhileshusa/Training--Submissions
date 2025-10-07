#Optimized Linear Regression Example
import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    # Use NumPy arrays efficiently (float32 saves memory)
    X = np.array([[1000], [1500], [2000], [2500], [3000]], dtype=np.float32)
    y = np.array([200, 250, 300, 350, 400], dtype=np.float32)

    # Initialize model once
    model = LinearRegression(n_jobs=-1)  # Use all CPU cores for performance

    # Train the model
    model.fit(X, y)

    # Vectorized prediction (faster than loops)
    new_sizes = np.array([[2200], [2700], [3200]], dtype=np.float32)
    predictions = model.predict(new_sizes) * 1000

    # Display results neatly
    for size, price in zip(new_sizes.flatten(), predictions):
        print(f"🏠 Size: {int(size)} sqft → Predicted Price: ${price:,.2f}")

if __name__ == "__main__":
    main()
