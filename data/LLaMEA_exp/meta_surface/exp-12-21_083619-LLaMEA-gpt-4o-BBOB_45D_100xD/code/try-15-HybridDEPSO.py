import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.F = 0.8
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.7
        self.elite_reinit_factor = 0.05  # Proportion of budget for elite reinitialization
        self.reinit_threshold = 0.2  # Threshold for reinitialization based on improvement

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, population.shape)
        
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        elite = global_best
        last_global_best_fitness = global_best_fitness

        while evaluations < self.budget:
            # Differential Evolution Phase with Dynamic Neighborhood
            for i in range(self.population_size):
                neighbors_idx = np.random.choice(range(self.population_size), 5, replace=False)
                a, b, c = population[neighbors_idx[:3]]
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

            # Particle Swarm Optimization Phase with Elite Reinitialization
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.inertia_weight * velocities[i] +
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

            # Update elite and adapt F
            if global_best_fitness < func(elite):
                elite = global_best
            self.F = 0.6 + 0.2 * (evaluations / self.budget)

            # Reinitialize elite if improvement stalls
            if (last_global_best_fitness - global_best_fitness) / last_global_best_fitness < self.reinit_threshold:
                if evaluations + int(self.elite_reinit_factor * self.budget) <= self.budget:
                    elite = np.random.uniform(lb, ub, self.dim)
                    last_global_best_fitness = global_best_fitness

        return elite, func(elite)