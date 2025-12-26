import numpy as np

class EnhancedDEWithDynamicScaling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)

            diversity_factor = np.mean(np.std(population, axis=0))
            dynamic_scaling_factor = self.base_mutation_factor * (1 + diversity_factor)
            adaptive_crossover_rate = self.base_crossover_rate * (1 - diversity_factor)

            mutant_vector = population[a] + dynamic_scaling_factor * (population[b] - population[c])
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
        return new_population, fitness

    def adaptive_local_search(self, candidate, candidate_fitness, func, bounds):
        for _ in range(5):  # Increase local search intensity
            step_size = (bounds.ub - bounds.lb) * 0.01
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
            new_fitness = func(new_candidate)
            if new_fitness < candidate_fitness:
                candidate, candidate_fitness = new_candidate, new_fitness
        return candidate, candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            population, fitness = self.differential_evolution(population, fitness, func)
            evaluations += self.population_size

            best_idx = np.argmin(fitness)
            best_candidate, best_fitness = population[best_idx], fitness[best_idx]

            if evaluations + 5 <= self.budget:
                best_candidate, best_fitness = self.adaptive_local_search(best_candidate, best_fitness, func, bounds)
                if best_fitness < fitness[best_idx]:
                    population[best_idx] = best_candidate
                    fitness[best_idx] = best_fitness

            evaluations += 5
            self.temperature *= self.cooling_rate

        return population[np.argmin(fitness)]