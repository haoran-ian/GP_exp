import numpy as np

class MetaPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.initial_F = 0.5
        self.initial_CR = 0.9
        self.elite_ratio = 0.1  # Percentage of elite individuals
        self.elite_size = int(self.population_size * self.elite_ratio)
        self.dynamic_factor = 0.99  # Dynamic factor for parameter adjustment

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)

            # Evaluate new positions and apply elitism
            fitness = np.array([func(ind) for ind in pop])
            evaluations += self.population_size
            ranked_indices = np.argsort(fitness)
            elite_indices = ranked_indices[:self.elite_size]
            if func(pop[elite_indices[0]]) < func(global_best):
                global_best = pop[elite_indices[0]]
            
            # Update personal bests
            better_indices = fitness < personal_best_fitness
            personal_best[better_indices] = pop[better_indices]
            personal_best_fitness[better_indices] = fitness[better_indices]

            # Differential Evolution step with dynamic parameters
            F = self.initial_F * self.dynamic_factor**(evaluations / self.budget)
            CR = self.initial_CR * self.dynamic_factor**(evaluations / self.budget)
            
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