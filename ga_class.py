import numpy as np

from genetic_algorithm.population import Population
from sklearn.metrics import mean_squared_error


class GeneticAlgorithm:
    """
    Genetic Algorithm for find the optimal window: (a, b)
    """
    def __init__(self, data_signal_1, data_signal_2, max_pop):
        self.data_signal_1 = data_signal_1
        self.data_signal_2 = data_signal_2

        self.genetic_algorithm = Population(max_pop=max_pop,
                                            low=0,
                                            high=self.data_signal_2.shape[0],
                                            fitness_calculator=self.fitness_calculator)

    def fitness_calculator(self, genes):
        point_a = genes[0]
        point_b = genes[1]

        signal_1 = self.data_signal_1[point_a:point_b + 1]
        signal_2 = self.data_signal_2[point_a:point_b + 1]

        return mean_squared_error(signal_1, signal_2)

    def run(self,
            eps=0.01,
            stop_criteria_samples=20,
            max_iteration=1000,
            mutation_rate=0.1,
            mutation_tolerance=10,
            mutation_range=range(-5, 5)):

        opt_window = None
        opt_loss = None

        avg_fitness_hist = list()
        best_window_hist = list()
        best_windows_performance = dict()

        i = 0
        is_finished = False
        while not (is_finished or i == max_iteration):
            self.genetic_algorithm.natural_selection()
            self.genetic_algorithm.generate(mutation_rate=mutation_rate,
                                            tolerance=mutation_tolerance,
                                            mutation_range=mutation_range)
            self.genetic_algorithm.eval_population()
            self.genetic_algorithm.validate()

            generation = self.genetic_algorithm.get_generation()
            best_window = self.genetic_algorithm.get_best()
            avg_fitness = self.genetic_algorithm.get_avg_fitness()

            avg_fitness_hist.append(avg_fitness)
            best_window_hist.append(best_window[0])
            best_windows_performance[best_window[0]] = best_window[1]

            print(f"Generation: {generation} | Best Window: {best_window[0]} | Average Loss: {avg_fitness}")

            if i > stop_criteria_samples:
                values = avg_fitness_hist[-stop_criteria_samples:]
                if np.max(values) - np.min(values) < eps:
                    opt_window = min(best_windows_performance, key=best_windows_performance.get)
                    print("Find local optima!")
                    print(f"Optimal window: {opt_window}. Loss: {best_windows_performance[opt_window]}.")
                    is_finished = True

            if i == max_iteration:
                opt_window = max(best_windows_performance, key=best_windows_performance.get)
                print("Maximum iteration reached!")
                print(f"Optimal window: {opt_window}. Loss: {best_windows_performance[opt_window]}.")

            i += 1

        return opt_window, opt_loss, best_window_hist, avg_fitness_hist
