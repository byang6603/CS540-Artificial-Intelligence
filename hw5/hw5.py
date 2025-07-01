import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    with open(filename, "r", newline='', encoding="utf-8") as file:
        csv_data = csv.DictReader(file)  # Pass file, not filename
        data = [dict(pair) for pair in csv_data]
    return data


def visualize_data(data):
    # Convert "year" and "days" to integers before plotting
    x = [int(item.get("year", 0)) for item in data]  # Default to 0 if key is missing
    y = [int(item.get("days", 0)) for item in data]  # Default to 0 if key is missing

    plt.plot(x, y)
    plt.title("Ice data")
    plt.xlabel("Year")
    plt.ylabel("Ice Days")
    plt.xlim(1854, 2024)
    plt.savefig("data_plot.jpg")


def normalize_data(data):
    years = [int(item.get("year", 0)) for item in data]  # Default to 0 if key is missing
    smallest = min(years)
    largest = max(years)

    normalized = [(x - smallest) / (largest - smallest) for x in years]

    x_normalized = np.column_stack((normalized, np.ones(len(normalized))))

    print("Q2:")
    print(x_normalized)

    y = [int(item.get("days", 0)) for item in data]  # Convert days to integer
    return np.column_stack((normalized, np.ones(len(normalized)))), y


def closed_form_solution(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


def gradient_descent(X, Y, learning_rate, iterations):
    n = len(Y)
    weights = np.array([0.0, 0.0])

    print("Q4a:")
    losses = {}
    for i in range(iterations):
        if i % 10 == 0:
            print(weights)
        predictions = X @ weights
        gradient = 1 / n * X.T @ (predictions - Y)
        weights -= learning_rate * gradient
        losses[i] = (weights[1])

    print("Q4b: 0.0001")
    print("Q4c: 100")
    print("Q4d: I got to these value by trying to tweak the learning rate first and then by changing the number of iterations. I found that small learning rate was the best way to achieve convergence even though it takes a long time to properly learn. I find that it is better to reach convergence than to not converge at all.")

    times = [iteration for iteration in losses.keys()]
    loss = [difference for difference in losses.values()]
    plt.plot(times, loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.xlim(1854, 2024)
    plt.savefig("loss_plot.jpg")

def prediction(data, weights, x):
    years = [int(item.get("year", 0)) for item in data]  # Default to 0 if key is missing
    smallest = min(years)
    largest = max(years)

    y_hat = weights[0] * (x - smallest) / (largest - smallest) + weights[1]
    return y_hat

def interpretation(weights):
    if weights[0] > 0: return ">"
    if weights[0] < 0: return "<"
    if weights[0] == 0: return "="

def no_ice_year(weights, data):
    years = [int(item.get("year", 0)) for item in data]  # Default to 0 if key is missing
    smallest = min(years)
    largest = max(years)

    return (-weights[1] * (largest - smallest)) / weights[0] + smallest

if __name__ == "__main__":
    filename = sys.argv[1]
    learning_rate = float(sys.argv[2])  # Ensure learning rate is a float
    iterations = int(sys.argv[3])  # Ensure iterations is an integer

    data = load_data(filename)
    visualize_data(data)

    normalized_data, y = normalize_data(data)

    weights = closed_form_solution(normalized_data, y)
    print("Q3:")
    print(weights)

    gradient_descent(normalized_data, y, learning_rate, iterations)

    y_hat = prediction(data, weights, 2023) #hard coded
    print("Q5: " + str(y_hat))

    symbol = interpretation(weights)
    print("Q6a: " + symbol)
    print("Q6b: A positive w means that as the years go on, the number of ice days tends to increase each year. A negative w means that as the years go on, the number of ice days tends to decrease each year. A w of 0 means that there is no trend between the years and the number of ice days that occurs in that year.")

    x_star = no_ice_year(weights, data)
    print("Q7a: " + str(x_star))
    print("Q7b: The answer from the calculations is compelling since within the near future, the world will get to the point where there will no longer be ice on Lake Mendota. From an environmental standpoint this is a very big concern. One limitation to this calculation can be how much global warming can worsen or change between now and the year 2400. Without considering this, our calculations can be off from the actual year that Lake Mendota will no longer have ice.")

