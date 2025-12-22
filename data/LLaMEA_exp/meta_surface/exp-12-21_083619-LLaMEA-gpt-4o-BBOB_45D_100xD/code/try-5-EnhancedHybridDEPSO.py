import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30
        self.min_population_size = 10
        self.F = 0.8
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population_size = self.initial_population_size
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

        while evaluations < self.budget:
            # Adjust population size dynamically
            population_size = int(self.initial_population_size * (1 - evaluations / self.budget)) + self.min_population_size
            population = population[:population_size]
            velocities = velocities[:population_size]
            fitness = fitness[:population_size]
            personal_best = personal_best[:population_size]
            personal_best_fitness = personal_best_fitness[:population_size]

            # Adaptive inertia weight
            inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            # Differential Evolution Phase
            for i in range(population_size):
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                
                crossover = np.random.rand(self.dim) < self.CR
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

            # Particle Swarm Optimization Phase
            for i in range(population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - population[i]) +
                                 self.c2 * r2 * (global_best - population[i]))
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