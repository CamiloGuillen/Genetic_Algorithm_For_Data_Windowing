import copy
import numpy as np


class DNA:
    """
    Class to generate genes of the type (a, b)
    """
    def __init__(self, low=0, high=150):
        self.fitness = 0
        self.low = low
        self.high = high
        self.genes = self.genes_generator(self.low, self.high)

    @staticmethod
    def genes_generator(low=0, high=150):
        a = None
        b = None
        valid_gene = False

        while not valid_gene:
            a = np.random.randint(low=low, high=high)
            b = np.random.randint(low=low, high=high)

            if a <= b:
                valid_gene = True

        return [a, b]

    def calc_fitness(self, fitness_calculator=None):
        self.fitness = fitness_calculator(self.genes)

        return None

    def reproduce(self, parent):
        child = DNA(low=self.low, high=self.high)

        eps = np.random.random()

        if eps > 0.5:
            if self.genes[0] <= parent.genes[1]:
                child.genes = [self.genes[0], parent.genes[1]]
            else:
                eps_ = np.random.random()
                if eps_ > 0.5:
                    child.genes = self.genes
                else:
                    child.genes = parent.genes

        else:
            if parent.genes[0] <= self.genes[1]:
                child.genes = [parent.genes[0], self.genes[1]]
            else:
                eps_ = np.random.random()
                if eps_ > 0.5:
                    child.genes = self.genes
                else:
                    child.genes = parent.genes

        return child

    def mutate(self, mutation_rate, tolerance, mutation_range):
        for i, _ in enumerate(self.genes):
            eps = np.random.random()

            if eps < mutation_rate:
                j = 0
                while j < tolerance:
                    mutate_value = np.random.choice(list(mutation_range))
                    mutate_gene = copy.deepcopy(self.genes)
                    mutate_gene[i] += mutate_value

                    if self.valid_gene(mutate_gene):
                        self.genes[i] = mutate_gene[i]
                        break
                    else:
                        j += 1

        return None

    @staticmethod
    def valid_gene(gene):
        if gene[1] >= gene[0] >= 0 and gene[1] >= 0:
            return True
        else:
            return False
