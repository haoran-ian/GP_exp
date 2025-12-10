import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Size of the population
        self.inertia_weight = 0.7  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.F_min, self.F_max = 0.4, 0.9  # Adaptive DE scaling factor range
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = func(global_best)
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Update velocities and positions for PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

            # Evaluate new positions
            for i in range(self.population_size):
                new_fitness = func(pop[i])
                evaluations += 1
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < global_best_fitness:
                        global_best = pop[i]
                        global_best_fitness = new_fitness
                        if evaluations >= self.budget:
                            break

            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                F = np.random.uniform(self.F_min, self.F_max)
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness
                        if evaluations >= self.budget:
                            break
        
        return global_best