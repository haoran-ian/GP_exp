import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Increased population size
        self.F = 0.8  # Adaptive DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.c1 = 2.0  # PSO cognitive parameter
        self.c2 = 2.0  # PSO social parameter
        self.inertia_weight = 0.7  # Inertia weight for PSO

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
        elite = global_best  # Elite preservation

        while evaluations < self.budget:
            # Differential Evolution Phase
            for i in range(self.population_size):
                indices = list(range(self.population_size))
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

            # Adaptive local search
            if evaluations < self.budget - 2:
                local_search = np.clip(global_best + 0.1 * np.random.randn(self.dim), lb, ub)
                local_fitness = func(local_search)
                evaluations += 1
                if local_fitness < global_best_fitness:
                    global_best = local_search
                    global_best_fitness = local_fitness

            # Update elite and adapt F
            if global_best_fitness < func(elite):
                elite = global_best
            self.F = 0.5 + 0.3 * (evaluations / self.budget)

        return elite, func(elite)