import numpy as np

class EnhancedAdaptiveStochasticSearchAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.min_mutation_strength = 0.01
        self.max_mutation_strength = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.random_state.uniform(lb, ub, size=(self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]
            
            # Introduce diversity in the elite
            elite = self.introduce_elite_diversity(elite, lb, ub)

            offspring = []
            for _ in range(self.population_size - elite_count):
                parent = elite[self.random_state.randint(elite_count)]
                offspring.append(self.adaptive_mutate(parent, lb, ub, evaluations))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

        best_index = np.argmin(fitness)
        return population[best_index]

    def adaptive_mutate(self, parent, lb, ub, evaluations):
        # Adaptive mutation strength based on the remaining budget
        progress = evaluations / self.budget
        mutation_strength = self.min_mutation_strength + (self.max_mutation_strength - self.min_mutation_strength) * (1 - progress)
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = parent + noise
        return np.clip(mutant, lb, ub)
    
    def introduce_elite_diversity(self, elite, lb, ub):
        # Add random perturbations to a fraction of elite solutions
        diversity_factor = 0.05
        for i in range(len(elite)):
            if self.random_state.rand() < diversity_factor:
                noise = self.random_state.normal(0, 0.05, size=self.dim)
                elite[i] = np.clip(elite[i] + noise, lb, ub)
        return elite