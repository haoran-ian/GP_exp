import numpy as np

class EnhancedAdaptivePopulationStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.7  # Initial Crossover probability
        self.mutation_strength = 0.1  # Initial mutation strength

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = self.initial_population_size
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        prev_best_fitness = np.inf

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub)
                offspring.append(self.mutate(child, lb, ub))

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))
            
            # Dynamically adjust population size
            population_size = max(10, int(self.initial_population_size * (1 - evaluations/self.budget)))
            
            # Adjust mutation and crossover rates based on convergence speed
            current_best_fitness = np.min(fitness)
            convergence_speed = abs(prev_best_fitness - current_best_fitness)
            self.mutation_strength = 0.1 * (1 - convergence_speed)
            self.cr = 0.7 + 0.3 * convergence_speed
            prev_best_fitness = current_best_fitness

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub):
        noise = self.random_state.normal(0, self.mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)