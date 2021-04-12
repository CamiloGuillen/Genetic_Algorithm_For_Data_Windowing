import numpy as np

from genetic_algorithm.dna import DNA
from scipy.special import softmax


class Population:
    """
    Class to generate a population with genes of the type (a, b)
    """
    def __init__(self, max_pop, low, high, fitness_calculator):
        self.generation = 0
        self.best = None
        self.pop_probabilities = None
        self.fitness_calculator = fitness_calculator
        self.low = low
        self.high = high

        self.population = list()
        for _ in range(max_pop):
            new_dna = DNA(low=self.low, high=self.high)
            self.population.append(new_dna)

        self.eval_population()

    def eval_population(self):
        for pop in self.population:
            pop.calc_fitness(fitness_calculator=self.fitness_calculator)

        return None

    def natural_selection(self):
        self.pop_probabilities = dict()
        all_fitness = [pop.fitness for pop in self.population]
        all_fitness = np.max(all_fitness) - all_fitness
        probabilities = softmax(all_fitness)

        for i, pop in enumerate(self.population):
            self.pop_probabilities[pop] = probabilities[i]

        return None

    def generate(self, mutation_rate, mutation_range, tolerance):
        for i in range(len(self.population)):
            indexes = list(range(len(self.population)))

            index_a = np.random.choice(a=indexes, p=list(self.pop_probabilities.values()))
            index_b = np.random.choice(a=indexes, p=list(self.pop_probabilities.values()))

            parent_a = self.population[index_a]
            parent_b = self.population[index_b]

            child = parent_a.reproduce(parent_b)

            child.mutate(mutation_rate=mutation_rate, tolerance=tolerance, mutation_range=mutation_range)

            self.population[i] = child

        self.generation += 1

        return None

    def validate(self):
        best_fitness = 1000000
        index = 0

        for i, pop in enumerate(self.population):
            if pop.fitness < best_fitness:
                index = i
                best_fitness = pop.fitness

        self.best = self.population[index]

        return None

    def get_best(self):

        return tuple(self.best.genes), self.best.fitness

    def get_generation(self):

        return self.generation

    def get_avg_fitness(self):
        total = 0

        for pop in self.population:
            total += pop.fitness

        return total/len(self.population)
