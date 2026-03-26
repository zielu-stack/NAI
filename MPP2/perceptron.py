from random import uniform

class Perceptron:
    def __init__(self, n, threshold=0, alpha=0.01):
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

def get_range(dataset):
    row_length = len(dataset[0]) - 1
    mins = [float('inf')] * row_length
    maxs = [float('inf') * -1] * row_length
    for row in dataset:
        for i in range(row_length):
            if row[i] < mins[i]:
                mins[i] = row[i]
            if row[i] > maxs[i]:
                maxs[i] = row[i]
    return [mins, maxs]

def normalize_dataset(dataset, mins, maxs):
    row_length = len(dataset[0]) - 1
    for row in dataset:
        for i in range(row_length):
            row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i])
        if row[-1] == "Iris-setosa":
            row[-1] = 1
        else:
            row[-1] = 0
    return dataset


training_dataset = create_dataset("..\\iris-data\\iris_training.txt")
test_dataset = create_dataset("..\\iris-data\\iris_test.txt")

mins, maxs = get_range(training_dataset)

training_dataset = normalize_dataset(training_dataset, mins, maxs)
test_dataset = normalize_dataset(test_dataset, mins, maxs)

perceptron = Perceptron(n=len(training_dataset[0])-1)

epochs = int(input("Ile epok? "))

for _ in range(epochs):
    for training_observation in training_dataset:
        perceptron.learn(training_observation[:-1], training_observation[-1])

while True:
    mode = int(input("Wybierz tryb:\n"
                 "0 - zakończ\n"
                 "1 - perceptron dla zbioru testowego\n"
                 "2 - perceptron dla recznie podanych danych\n"))
    if mode == 0:
        break
    elif mode == 1:
        correct = 0
        for test_observation in test_dataset:
            outcome = perceptron.compute_outcome(test_observation[:-1])
            if outcome == test_observation[-1]:
                correct += 1
        accuracy = correct / len(test_dataset)
        print(f"Dokladnosc: {accuracy*100}%")
    elif mode == 2:
        vector = list(map(float, input("Podaj wektor obserwacji (wartości oddzielone spacjami): ").split()))
        outcome = perceptron.compute_outcome(normalize_dataset([vector], mins, maxs)[0])
        print(f"Decyzja: {"Iris setosa" if outcome == 1 else "Nie-setosa"})")
