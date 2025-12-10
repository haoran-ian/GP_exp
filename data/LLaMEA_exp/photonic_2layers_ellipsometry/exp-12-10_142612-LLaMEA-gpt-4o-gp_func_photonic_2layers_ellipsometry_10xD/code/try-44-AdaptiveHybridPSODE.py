import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Size of the population
        self.inertia_weight_start = 0.9
        self.inertia_weight_end = 0.4
        self.c1 = 2.0  # Cognitive coefficient for PSO
        self.c2 = 2.0  # Social coefficient for PSO
        self.F_start = 0.5  # Initial DE scaling factor
        self.F_end = 0.9  # Final DE scaling factor
        self.CR_start = 0.8  # Initial DE crossover probability
        self.CR_end = 0.4  # Final DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Calculate dynamic parameters
            progress = evaluations / self.budget
            inertia_weight = self.inertia_weight_start * (1 - progress) + self.inertia_weight_end * progress
            F = self.F_start * (1 - progress) + self.F_end * progress
            CR = self.CR_start * (1 - progress) + self.CR_end * progress

            # Update velocities and positions for PSO
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (inertia_weight * velocity[i] +
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
                    if new_fitness < func(global_best):
                        global_best = pop[i]
                        if evaluations >= self.budget:
                            break

            # Differential Evolution step
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < func(global_best):
                        global_best = trial
                        if evaluations >= self.budget:
                            break

        return global_best