import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)  # Adjust population size based on budget
        self.mutation_factor = 0.8  # Differential Evolution mutation factor
        self.crossover_rate = 0.7  # Differential Evolution crossover rate
        self.temperature = 1.0  # Initial temperature for Simulated Annealing
        self.cooling_rate = 0.99  # Cooling rate for Simulated Annealing

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        for i in range(self.population_size):
            # Mutation
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
            # Crossover
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < self.crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            # Selection
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
        return new_population, fitness

    def simulated_annealing(self, candidate, candidate_fitness, func, bounds):
        perturbation = np.random.normal(0, 1, self.dim) * (bounds.ub - bounds.lb) * 0.1
        new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
        new_fitness = func(new_candidate)
        if (new_fitness < candidate_fitness or
            np.random.rand() < np.exp((candidate_fitness - new_fitness) / self.temperature)):
            return new_candidate, new_fitness
        return candidate, candidate_fitness

    def __call__(self, func):
        bounds = func.bounds
        # Initialize population
        population = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Apply Differential Evolution
            population, fitness = self.differential_evolution(population, fitness, func)
            evaluations += self.population_size

            # Apply Simulated Annealing to best candidate
            best_idx = np.argmin(fitness)
            best_candidate, best_fitness = population[best_idx], fitness[best_idx]
            new_candidate, new_fitness = self.simulated_annealing(best_candidate, best_fitness, func, bounds)

            if new_fitness < best_fitness:
                population[best_idx] = new_candidate
                fitness[best_idx] = new_fitness

            evaluations += 1  # Increment for SA evaluation
            self.temperature *= self.cooling_rate  # Cool down

        return population[np.argmin(fitness)]

# Example usage:
# optimizer = HybridDESA(budget=1000, dim=10)
# best_solution = optimizer(func)