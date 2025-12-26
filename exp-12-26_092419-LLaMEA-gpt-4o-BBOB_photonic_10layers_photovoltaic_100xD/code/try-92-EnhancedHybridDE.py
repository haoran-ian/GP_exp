import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.base_mutation_factor = 0.8
        self.base_crossover_rate = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.95  # Faster cooling rate for more rapid adaptation

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        diversity = np.mean(np.std(population, axis=0))
        adaptive_mutation_factor = self.base_mutation_factor * (1 + 0.5 * diversity)
        adaptive_crossover_rate = self.base_crossover_rate * (1 - 0.5 * diversity)
        
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

    def stochastic_tunneling(self, candidate, candidate_fitness, func, bounds):
        perturbation = np.random.normal(0, 1, self.dim) * (bounds.ub - bounds.lb) * 0.05
        new_candidate = np.clip(candidate + perturbation, bounds.lb, bounds.ub)
        new_fitness = func(new_candidate)
        if new_fitness < candidate_fitness or np.random.rand() < np.exp((candidate_fitness - new_fitness) / (self.temperature * 0.1)):
            return new_candidate, new_fitness
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
            new_candidate, new_fitness = self.stochastic_tunneling(best_candidate, best_fitness, func, bounds)

            if new_fitness < best_fitness:
                population[best_idx] = new_candidate
                fitness[best_idx] = new_fitness

            evaluations += 1
            self.temperature *= self.cooling_rate

        return population[np.argmin(fitness)]