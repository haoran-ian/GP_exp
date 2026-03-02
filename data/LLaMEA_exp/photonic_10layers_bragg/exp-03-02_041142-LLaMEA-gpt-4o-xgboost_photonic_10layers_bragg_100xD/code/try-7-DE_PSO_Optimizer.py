import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.c1 = 2.0  # PSO cognitive coefficient
        self.c2 = 2.0  # PSO social coefficient
        self.w = 0.9  # Adaptive inertia weight
        self.vel_limit = 0.1  # Velocity limit factor
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        vel = np.zeros((self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.pop_size
        
        while evaluations < self.budget:
            # DE mutation and crossover
            for i in range(self.pop_size):
                indices = list(range(i)) + list(range(i + 1, self.pop_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), lb, ub)
                self.CR = 0.5 + 0.5 * np.random.rand()  # Dynamic crossover adjustment
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                trial_value = func(trial)
                evaluations += 1
                
                if trial_value < personal_best_values[i]:
                    personal_best[i], personal_best_values[i] = trial, trial_value
                    if trial_value < global_best_value:
                        global_best, global_best_value = trial, trial_value
            
            # PSO velocity and position update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = self.w * vel[i] + self.c1 * r1 * (personal_best[i] - pop[i]) + self.c2 * r2 * (global_best - pop[i])
                vel[i] = np.clip(vel[i], -self.vel_limit * (ub - lb), self.vel_limit * (ub - lb))
                pop[i] = np.clip(pop[i] + vel[i], lb, ub)
                
                # Evaluate new solutions and update personal best
                new_value = func(pop[i])
                evaluations += 1
                if new_value < personal_best_values[i]:
                    personal_best[i], personal_best_values[i] = pop[i], new_value
                    if new_value < global_best_value:
                        global_best, global_best_value = pop[i], new_value
                        
        return global_best