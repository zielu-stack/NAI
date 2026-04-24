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
        chars = {chr(i):0 for i in range(97, 123)}
        for line in file:
            for char in line:
                char = char.lower()
                if char in chars:
                    chars[char] += 1
        total = sum(chars.values())
        chars = {i:chars[i]/total for i in chars}
        return chars

def create_layer(k=3, n=26, default_threshold=0, default_alpha=0.01):
        layer = [Perceptron(n, default_threshold, default_alpha) for _ in range(k)]

