import numpy as np

class AdaptiveHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.98  # Change 1: Slightly increased cooling rate for SA

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        diversity = np.mean(np.std(population, axis=0))
        fitness_variance = np.std(fitness) / (np.mean(fitness) + 1e-7)  # Change 2
        adaptive_mutation_factor = self.mutation_factor * (1 + 0.5 * diversity + 0.5 * fitness_variance)  # Change 3
        adaptive_crossover_rate = self.crossover_rate * (1 - 0.5 * diversity - 0.5 * fitness_variance)  # Change 4
        
        inertia_weight = 0.9 - (0.9 - 0.4) * (fitness - min(fitness)) / (max(fitness) - min(fitness) + 1e-7)

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = (1 - inertia_weight[i]) * population[i] + inertia_weight[i] * (population[a] + adaptive_mutation_factor * (population[b] - population[c]))
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
        return new_population, fitness

    def simulated_annealing(self, candidate, candidate_fitness, func, bounds):
        perturbation = np.random.normal(0, 1, self.dim) * (bounds.ub - bounds.lb) * 0.1
        new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
        new_fitness = func(new_candidate)
        if new_fitness < candidate_fitness or np.random.rand() < np.exp((candidate_fitness - new_fitness) / self.temperature):
            return new_candidate, new_fitness
        return candidate, candidate_fitness

    def local_search(self, candidate, candidate_fitness, func, bounds):
        step_size = (bounds.ub - bounds.lb) * 0.01
        for _ in range(10):
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
            new_candidate, new_fitness = self.simulated_annealing(best_candidate, best_fitness, func, bounds)

            if new_fitness < best_fitness:
                population[best_idx] = new_candidate
                fitness[best_idx] = new_fitness

            evaluations += 1
            self.temperature *= self.cooling_rate

            if evaluations + 10 <= self.budget:
                best_candidate, best_fitness = self.local_search(best_candidate, best_fitness, func, bounds)
                if best_fitness < fitness[best_idx]:
                    population[best_idx] = best_candidate
                    fitness[best_idx] = best_fitness
                evaluations += 10

        return population[np.argmin(fitness)]