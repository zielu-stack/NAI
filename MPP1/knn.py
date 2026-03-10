import math

def create_dataset(filename):
    with open(filename, "r") as file:
        raw_dataset = [line.strip().split() for line in file.readlines()]
        parsed_dataset = []
        for raw_row in raw_dataset:
            parsed_row = []
            for i in range(0, len(raw_row)-1):
                number = raw_row[i].replace(",", ".")
                parsed_row.append(float(number))
            parsed_row.append(raw_row[len(raw_row)-1])
            parsed_dataset.append(parsed_row)
        return parsed_dataset

def knn(training_data, test_observation, k):
    best_k_deltas = [float("inf")] * k
    best_k_results = [None] * k
    for vector in training_data:
        distance_sum = 0
        for j in range(len(vector)-1):
            distance_sum += (test_observation[j] - vector[j]) ** 2
        max_delta = max(best_k_deltas)
        if distance_sum < max_delta:
            index = best_k_deltas.index(max_delta)
            best_k_results[index] = vector[-1]
            best_k_deltas[index] = distance_sum
    results_summary = {key:[0, 0.0] for key in set(best_k_results)}      #counter & sum of deltas
    for i in range(k):
        results_summary[best_k_results[i]][0] += 1
        results_summary[best_k_results[i]][1] += best_k_deltas[i]
    best_result = None
    max_counter = 0
    min_avg_delta = float("inf")
    for result in results_summary:
        counter = results_summary[result][0]
        avg_delta = results_summary[result][1] / results_summary[result][0]
        if counter > max_counter or (counter == max_counter and avg_delta < min_avg_delta):
            best_result = result
            max_counter = counter
            min_avg_delta = avg_delta
    return best_result

training_dataset = create_dataset("iris_training.txt")
test_dataset = create_dataset("iris_test.txt")

for k in range(1, 14):
    correct_counter = 0
    for observation in test_dataset:
        knn_outcome = knn(training_dataset, observation, k)
        if knn_outcome == observation[-1]:
            correct_counter += 1
        else:
            print(f"KNN: {knn_outcome}, poprawna odpowiedź: {observation[-1]}")
    print(f"Dokładność: {correct_counter/len(test_dataset)*100}%, {k}")
