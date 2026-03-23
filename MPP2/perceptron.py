from random import random

class Perceptron:
    def __init__(self, n, ):
        self.weights = [rand for _ in range(n)]
        self.threshold = 0
        return

    def learn(self, dataset, epoch, ):
        return

def create_dataset(filename):
    with open(filename, "r") as file:
        raw_dataset = [line.strip().split() for line in file.readlines()]
        parsed_dataset = []
        row_length = len(raw_dataset[0]) - 1
        for raw_row in raw_dataset:
            parsed_row = []
            for i in range(0, row_length):
                number = raw_row[i].replace(",", ".")
                parsed_row.append(float(number))
            parsed_row.append(raw_row[row_length])
            parsed_dataset.append(parsed_row)
        return parsed_dataset


training_dataset = create_dataset("iris_training.txt")
test_dataset = create_dataset("iris_test.txt")
