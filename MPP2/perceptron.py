from random import uniform

class Perceptron:
    def __init__(self, n, threshold=0.5, alpha=0.01):
        self.weights = [uniform(-1,1) for _ in range(n)]
        self.threshold = threshold
        self.alpha = alpha

    def compute_outcome(self, observation):
        net = sum(w * x for w,x in zip(self.weights,observation))
        if net >= self.threshold:
            return 1
        return 0

    def learn(self, observation, expected_outcome):
        outcome = self.compute_outcome(observation)
        diff = expected_outcome - outcome
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * diff * observation[i]
        self.threshold += self.alpha * diff * -1

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

training_dataset = create_dataset("..\\iris-data\\iris_training.txt")
test_dataset = create_dataset("..\\iris-data\\iris_test.txt")

perceptron = Perceptron(n=4)

epoch = int(input("How many epochs? "))

for _ in range(epoch):
    for training_observation in training_dataset:
        perceptron.learn(training_observation[:-1], training_observation[-1])


for test_observation in test_dataset:
    outcome = perceptron.compute_outcome(test_observation[:-1])

