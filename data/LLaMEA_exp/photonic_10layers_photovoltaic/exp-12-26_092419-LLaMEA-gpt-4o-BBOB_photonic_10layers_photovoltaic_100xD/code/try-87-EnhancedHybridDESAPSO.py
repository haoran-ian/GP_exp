import numpy as np

class EnhancedHybridDESAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.temperature = 1.0
        self.cooling_rate = 0.99
        self.velocities = np.zeros((self.population_size, self.dim))
        self.inertia_weight = 0.9
        self.c1 = 1.5
        self.c2 = 1.5

    def differential_evolution(self, population, fitness, func):
        new_population = np.copy(population)
        diversity = np.mean(np.std(population, axis=0))
        adaptive_mutation_factor = self.mutation_factor * (1 + diversity)
        adaptive_crossover_rate = self.crossover_rate * (1 - diversity)
        
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant_vector = population[a] + adaptive_mutation_factor * (population[b] - population[c])
            trial_vector = np.copy(population[i])
            crossover_points = np.random.rand(self.dim) < adaptive_crossover_rate
            trial_vector[crossover_points] = mutant_vector[crossover_points]
            trial_fitness = func(trial_vector)
            if trial_fitness < fitness[i]:
                new_population[i] = trial_vector
                fitness[i] = trial_fitness
        return new_population, fitness

    def particle_swarm_optimization(self, population, fitness, personal_best, personal_best_fitness, global_best):
        for i in range(self.population_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                  self.c1 * r1 * (personal_best[i] - population[i]) +
                                  self.c2 * r2 * (global_best - population[i]))
            population[i] = np.clip(population[i] + self.velocities[i], func.bounds.lb, func.bounds.ub)
            curr_fitness = func(population[i])
            if curr_fitness < personal_best_fitness[i]:
                personal_best[i] = population[i]
                personal_best_fitness[i] = curr_fitness
        return population, personal_best, personal_best_fitness

    def __call__(self, func):
        bounds = func.bounds
        population = np.random.rand(self.population_size, self.dim) * (bounds.ub - bounds.lb) + bounds.lb
        fitness = np.array([func(ind) for ind in population])
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best = population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            population, fitness = self.differential_evolution(population, fitness, func)
            evaluations += self.population_size

            global_best = population[np.argmin(fitness)]
            population, personal_best, personal_best_fitness = self.particle_swarm_optimization(population, fitness, personal_best, personal_best_fitness, global_best)
            evaluations += self.population_size

            if evaluations + 10 <= self.budget:
                best_idx = np.argmin(fitness)
                best_candidate, best_fitness = population[best_idx], fitness[best_idx]
                step_size = (bounds.ub - bounds.lb) * 0.01
                for _ in range(10):
                    perturbation = np.random.uniform(-step_size, step_size, self.dim)
                    new_candidate = np.clip(best_candidate + perturbation, bounds.lb, bounds.ub)
                    new_fitness = func(new_candidate)
                    if new_fitness < best_fitness:
                        best_candidate, best_fitness = new_candidate, new_fitness
                population[best_idx], fitness[best_idx] = best_candidate, best_fitness
                evaluations += 10

        return population[np.argmin(fitness)]