import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Step 0: New dataset
X = np.array([2, 4, 6, 8, 10, 12, 14]).reshape(-1, 1)  # Hours studied
y = np.array([30, 45, 55, 65, 70, 80, 85])             # Exam scores

# Create test points for smooth curve plotting
X_test = np.linspace(0, 15, 100).reshape(-1, 1)

# Function to plot model predictions
def plot_model(degree, title, subplot):
    # Step 1: Create polynomial features
    poly = PolynomialFeatures(degree=degree)           # Initialize polynomial features
    X_poly = poly.fit_transform(X)                     # Transform X into polynomial features

    # Step 2: Train Linear Regression model
    model = LinearRegression()                         # Initialize the model
    model.fit(X_poly, y)                               # Train the model

    # Step 3: Make predictions
    X_test_poly = poly.transform(X_test)               # Transform test points
    y_pred = model.predict(X_test_poly)                # Predict values

    # Step 4: Plot
    plt.subplot(1, 3, subplot)
    plt.scatter(X, y, color="red", label="Data Points")
    plt.plot(X_test, y_pred, color="blue", label=f"Degree {degree} Fit")
    plt.title(title)
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.legend()

def main():
    # Plot all three scenarios
    plt.figure(figsize=(15, 4))

    # Underfitting: Degree 1 (Linear)
    plot_model(degree=1, title="Underfitting (Linear)", subplot=1)

    # Good Fit: Degree 2 (Quadratic)
    plot_model(degree=2, title="Good Fit (Quadratic)", subplot=2)

    # Overfitting: Degree 6 (High Polynomial)
    plot_model(degree=6, title="Overfitting (6th Degree)", subplot=3)

    plt.tight_layout()
    plt.savefig("fitting_examples.png", dpi=300)  # Save as high-quality PNG
    plt.close()  # Close the figure to free memory
    print("Plots saved as fitting_examples.png")

if __name__ == "__main__":
    main()