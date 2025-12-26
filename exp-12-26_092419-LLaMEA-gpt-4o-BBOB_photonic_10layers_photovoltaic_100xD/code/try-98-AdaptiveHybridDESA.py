import numpy as np

class AdaptiveHybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        diversity = np.mean(np.std(population, axis=0))
        adaptive_mutation_factor = self.mutation_factor * (1 + diversity)
        adaptive_crossover_rate = self.dynamic_crossover_rate(diversity)
        
        inertia_weight = 0.9 - (0.9 - 0.4) * (fitness - min(fitness)) / (max(fitness) - min(fitness) + 1e-7)

        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = (1 - inertia_weight[i]) * population[i] + inertia_weight[i] * (population[a] + adaptive_mutation_factor * (population[b] - population[c]))
            mutant_vector = self.chaotic_perturbation(mutant_vector, fitness[i], func)
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
        return new_population, fitness

    def chaotic_perturbation(self, vector, current_fitness, func):
        chaos = np.sin(np.pi * vector)
        new_vector = np.clip(vector + chaos * 0.1, func.bounds.lb, func.bounds.ub)
        if func(new_vector) < current_fitness:
            return new_vector
        return vector

    def dynamic_crossover_rate(self, diversity):
        return 0.5 + 0.5 * np.tanh(10 * (diversity - 0.5))

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