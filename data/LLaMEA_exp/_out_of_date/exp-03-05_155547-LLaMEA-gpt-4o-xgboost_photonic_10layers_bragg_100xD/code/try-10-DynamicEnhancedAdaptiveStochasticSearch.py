import numpy as np

class DynamicEnhancedAdaptiveStochasticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.elite_fraction = 0.2
        self.random_state = np.random.RandomState(seed=42)
        self.cr = 0.9  # Crossover probability
        self.memory_factor = 0.1
        self.pop_size_factor = 0.05  # Adjust based on budget

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population_size = int(self.initial_population_size + self.pop_size_factor * self.budget)
        population = self.random_state.uniform(lb, ub, size=(population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        memory = population[np.argsort(fitness)[:int(self.memory_factor * population_size)]]

        while evaluations < self.budget:
            elite_count = int(self.elite_fraction * population_size)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite = population[elite_indices]

            offspring = []
            for _ in range(population_size - elite_count):
                parent1 = elite[self.random_state.randint(elite_count)]
                parent2 = elite[self.random_state.randint(elite_count)]
                child = self.crossover(parent1, parent2, lb, ub)
                child = self.mutate(child, lb, ub)
                
                # Memory-based exploration
                if self.random_state.rand() < self.memory_factor:
                    memory_ind = memory[self.random_state.randint(len(memory))]
                    child = self.memory_blend(child, memory_ind, lb, ub)
                
                offspring.append(child)

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            population = np.vstack((elite, offspring))
            fitness = np.concatenate((fitness[elite_indices], offspring_fitness))

            # Update memory
            combined = np.vstack((memory, population))
            combined_fitness = np.concatenate((fitness[:len(memory)], fitness))
            memory = combined[np.argsort(combined_fitness)[:int(self.memory_factor * population_size)]]

        best_index = np.argmin(fitness)
        return population[best_index]

    def crossover(self, parent1, parent2, lb, ub):
        mask = self.random_state.rand(self.dim) < self.cr
        child = np.where(mask, parent1, parent2)
        return np.clip(child, lb, ub)

    def mutate(self, individual, lb, ub):
        mutation_strength = self.random_state.rand() * 0.1
        noise = self.random_state.normal(0, mutation_strength, size=self.dim)
        mutant = individual + noise
        return np.clip(mutant, lb, ub)

    def memory_blend(self, child, memory_ind, lb, ub):
        blend_factor = self.random_state.rand()
        blended_child = blend_factor * child + (1 - blend_factor) * memory_ind
        return np.clip(blended_child, lb, ub)