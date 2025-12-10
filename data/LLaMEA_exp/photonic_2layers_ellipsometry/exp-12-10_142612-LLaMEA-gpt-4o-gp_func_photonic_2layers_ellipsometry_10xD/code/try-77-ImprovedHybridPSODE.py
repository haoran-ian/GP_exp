import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = np.random.uniform(0.4, 0.9)
        self.CR = 0.9
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            # Adaptive inertia weight
            self.inertia_weight = 0.4 + 0.5 * (1 - self.evaluations / self.budget)

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
                self.evaluations += 1
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < func(global_best):
                        global_best = pop[i]
                        if self.evaluations >= self.budget:
                            break

            # Differential Evolution step with diversity boost
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                perturbation = np.random.normal(0, 0.1, self.dim) * (ub - lb)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant + perturbation, pop[i])
                
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    if trial_fitness < func(global_best):
                        global_best = trial
                        if self.evaluations >= self.budget:
                            break

        return global_best