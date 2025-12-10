import numpy as np

class AdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.initial_F = 0.8
        self.final_F = 0.2
        self.C1 = 1.5
        self.C2 = 1.5
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        evaluations = self.population_size
        
        while evaluations < self.budget:
            progress = evaluations / self.budget
            inertia_weight = (self.initial_inertia_weight - self.final_inertia_weight) * (1 - progress) + self.final_inertia_weight
            F = (self.initial_F - self.final_F) * (1 - progress) + self.final_F

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (inertia_weight * velocity[i] +
                               self.C1 * r1 * (personal_best[i] - pop[i]) +
                               self.C2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

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

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), lb, ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                
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