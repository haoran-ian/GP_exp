import numpy as np

class EnhancedHybridDEPSOAdaptiveElitismV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_initial_size = 30
        self.population_variability = 5
        self.F_initial = 0.8
        self.CR_initial = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.velocity_clamp = np.array([2.0] * dim)  # Changed from 1.0 to 2.0

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.population_initial_size
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, population.shape)
        
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        elite = global_best

        def adaptive_inertia_weight(progress):
            return self.inertia_weight_initial - progress * (self.inertia_weight_initial - self.inertia_weight_final)

        def adaptive_parameters(progress):
            F = self.F_initial * (1 - progress)
            CR = self.CR_initial * (progress)
            return F, CR

        def dynamic_crowding_distance(pop, fit):
            distances = np.zeros(pop.shape[0])
            sorted_indices = np.argsort(fit)
            for i in range(self.dim):
                sorted_pop = pop[sorted_indices, i]
                max_val, min_val = np.max(sorted_pop), np.min(sorted_pop)
                if max_val == min_val:
                    continue
                distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
                for j in range(1, len(pop) - 1):
                    distances[sorted_indices[j]] += (sorted_pop[j + 1] - sorted_pop[j - 1]) / (max_val - min_val)
            return distances

        while evaluations < self.budget:
            if evaluations % (self.budget // 4) == 0:
                population_size = max(20, self.population_initial_size + np.random.randint(-self.population_variability, self.population_variability))
                if population_size > len(population):
                    new_individuals = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                    population = np.concatenate((population, new_individuals))
                    velocities = np.concatenate((velocities, np.random.uniform(-1, 1, new_individuals.shape)))
                    fitness = np.concatenate((fitness, [func(ind) for ind in new_individuals]))
                    personal_best = np.concatenate((personal_best, new_individuals))
                    personal_best_fitness = np.concatenate((personal_best_fitness, [func(ind) for ind in new_individuals]))
                    evaluations += len(new_individuals)
                else:
                    population = population[:population_size]
                    velocities = velocities[:population_size]
                    fitness = fitness[:population_size]
                    personal_best = personal_best[:population_size]
                    personal_best_fitness = personal_best_fitness[:population_size]

            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                progress = evaluations / self.budget
                F, CR = adaptive_parameters(progress)
                mutant = np.clip(a + F * (b - c), lb, ub)

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

            distances = dynamic_crowding_distance(population, fitness)
            for i in range(population_size):
                inertia_weight = adaptive_inertia_weight(evaluations / self.budget)
                if distances[i] < np.mean(distances):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (inertia_weight * velocities[i] +
                                     self.c1 * r1 * (personal_best[i] - population[i]) +
                                     self.c2 * r2 * (global_best - population[i]))
                    velocities[i] = np.clip(velocities[i], -self.velocity_clamp, self.velocity_clamp)
                    population[i] = np.clip(population[i] + velocities[i], lb, ub)

                    new_fitness = func(population[i])
                    evaluations += 1
                    if new_fitness < fitness[i]:
                        fitness[i] = new_fitness
                        if new_fitness < personal_best_fitness[i]:
                            personal_best[i] = population[i]
                            personal_best_fitness = new_fitness
                            if new_fitness < global_best_fitness:
                                global_best = population[i]
                                global_best_fitness = new_fitness

            if global_best_fitness < func(elite):
                elite = global_best

        for _ in range(5):  # Elitism-based local search
            perturbation = np.random.normal(0, 0.1, self.dim)
            candidate = np.clip(elite + perturbation, lb, ub)
            candidate_fitness = func(candidate)
            evaluations += 1
            if candidate_fitness < global_best_fitness:
                elite = candidate
                global_best_fitness = candidate_fitness

        return elite, func(elite)