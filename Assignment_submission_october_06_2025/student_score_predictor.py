# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    # Step 2: Create a small dataset
    data = {
        "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Exam_Score": [35, 40, 50, 55, 60, 65, 70, 75, 85, 90]
    }

    df = pd.DataFrame(data)
    print(" Dataset:")
    print(df)

    # Step 3: Prepare data for the model
    X = df[["Hours_Studied"]]   # Features (input)
    y = df["Exam_Score"]         # Target (output)

    # Step 4: Build the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Step 5: User input
    hours = float(input("\nEnter number of hours studied: "))

    # Step 6: Make prediction
    predicted_score = model.predict([[hours]])[0]

    # Step 7: Show results
    print(f"\n\ Predicted Exam Score for {hours} hours of study: {predicted_score:.2f}")

    # Step 8: Show model insights
    print("\nModel Insights:")
    print(f"➡ Coefficient (slope): {model.coef_[0]:.2f}")
    print(f"➡ Intercept: {model.intercept_:.2f}")

if __name__ == "__main__":
    main()
