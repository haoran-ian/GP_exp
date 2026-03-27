import numpy as np

class EnhancedGridDynamicStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Initial Crossover probability
        self.grid_divisions = 5  # Grid divisions for space partitioning

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            # Grid-based population management
            grid_size = (ub - lb) / self.grid_divisions
            grid_population = np.array([lb + self.random_state.randint(self.grid_divisions, size=self.dim) * grid_size for _ in range(population_size - elite_count)])
            offspring = []

            for i in range(population_size - elite_count):
                parent1_idx = self.random_state.choice(elite_count)
                parent2_idx = self.random_state.choice(elite_count)
                parent1, parent2 = elite[parent1_idx], elite[parent2_idx]
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness))
                offspring.append(self.mutate(child, lb, ub, np.std(fitness)))

            offspring = grid_population + np.array(offspring)  # Combine grid position with offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size and grid divisions
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))
            self.grid_divisions = max(2, int(5 * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, diversity):
        self.cr = 0.7 + 0.2 * diversity  # Adjust crossover probability based on diversity
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity):
        mutation_strength = self.random_state.rand() * 0.1 * diversity  # Adjust mutation strength based on diversity
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)