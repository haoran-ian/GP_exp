import numpy as np

class RefinedHybridDEPSOAdaptiveElitism:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F_initial = 0.8
        self.CR_initial = 0.9
        self.c1_initial = 2.0
        self.c2_initial = 2.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.velocity_clamp = np.array([1.0] * dim)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        personal_best = population.copy()
        personal_best_fitness = fitness.copy()

        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        elite = global_best

        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = population[indices]

                progress = evaluations / self.budget
                F = self.F_initial * (1 - progress)
                CR = self.CR_initial * progress
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, lb, ub)

                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness

            distances = self.dynamic_crowding_distance(population, fitness)
            inertia_weight = self.adaptive_inertia_weight(evaluations / self.budget)
            for i in range(self.population_size):
                if distances[i] < np.mean(distances):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (inertia_weight * velocities[i] +
                                     self.adaptive_c1(evaluations / self.budget) * r1 * (personal_best[i] - population[i]) +
                                     self.adaptive_c2(evaluations / self.budget) * r2 * (global_best - population[i]))
                    velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                    population[i] = np.clip(population[i] + velocities[i], lb, ub)

                    new_fitness = func(population[i])
                    evaluations += 1
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness
                        if new_fitness < personal_best_fitness[i]:
                            personal_best[i] = population[i]
                            personal_best_fitness[i] = new_fitness
                            if new_fitness < global_best_fitness:
                                global_best = population[i]
                                global_best_fitness = new_fitness

            if global_best_fitness < func(elite):
                elite = global_best

        return elite, func(elite)

    def adaptive_inertia_weight(self, progress):
        return self.inertia_weight_initial - progress * (self.inertia_weight_initial - self.inertia_weight_final)

    def adaptive_c1(self, progress):
        return self.c1_initial * (1 - progress)

    def adaptive_c2(self, progress):
        return self.c2_initial * progress

    def dynamic_crowding_distance(self, population, fitness):
        distances = np.zeros(population.shape[0])
        sorted_indices = np.argsort(fitness)
        for i in range(self.dim):
            sorted_pop = population[sorted_indices, i]
            max_val, min_val = np.max(sorted_pop), np.min(sorted_pop)
            if max_val == min_val:
                continue
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
            for j in range(1, len(population) - 1):
                distances[sorted_indices[j]] += (sorted_pop[j + 1] - sorted_pop[j - 1]) / (max_val - min_val)
        return distances