import numpy as np

class PhaseBasedAdaptiveStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Initial crossover probability
        self.phase_thresholds = [0.3, 0.6]  # Define phases based on budget consumption

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

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub, np.std(fitness))
                offspring.append(self.mutate(child, lb, ub, np.std(fitness)))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size and strategy based on phases
            phase_ratio = evaluations / self.budget
            if phase_ratio < self.phase_thresholds[0]:  # Exploration Phase
                self.cr = 0.9
            elif phase_ratio < self.phase_thresholds[1]:  # Balanced Phase
                self.cr = 0.7
            else:  # Exploitation Phase
                self.cr = 0.5

            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub, diversity):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub, diversity):
        mutation_strength = self.random_state.rand() * 0.1 * diversity
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)