
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([2, 4, 6, 8, 10, 12, 14]).reshape(-1, 1)
y = np.array([30, 45, 55, 65, 70, 80, 85])

X_test = np.linspace(0, 16, 100).reshape(-1, 1)


def plot_model(degree, title, subplot):

    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    model.fit(X_poly, y)

    X_test_poly = poly.transform(X_test)

    y_pred = model.predict(X_test_poly)

    plt.subplot(1, 3, subplot)

    plt.scatter(X, y, color='red', label='Data Points')

    plt.plot(X_test, y_pred, color='blue',
             label=f"Degree {degree} Fit")

    plt.title(title)

    plt.xlabel("Study Hours")

    plt.ylabel("Exam Score")

    plt.legend()


def main():

    plt.figure(figsize=(15, 4))

    plot_model(degree=1, title="Underfitting (Linear)", subplot=1)

    plot_model(degree=2, title="Good Fit (Quadratic)", subplot=2)

    plot_model(degree=6, title="Overfitting (6th Degree)", subplot=3)

    plt.tight_layout()

    plt.savefig("fitting_examples.png", dpi=300)

    plt.close()

    print("Plots saved as fitting_examples.png")


if __name__ == "__main__":
    main()

