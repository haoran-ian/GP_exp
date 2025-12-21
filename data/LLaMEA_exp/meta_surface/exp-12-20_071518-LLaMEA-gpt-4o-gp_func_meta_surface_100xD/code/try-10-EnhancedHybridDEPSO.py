import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Population size
        self.F = 0.5        # Initial Differential evolution scale factor
        self.CR = 0.9       # Crossover probability
        self.w = 0.5        # Inertia weight for PSO
        self.c1 = 2.0       # Cognitive weight
        self.c2 = 2.0       # Social weight
        self.eval_count = 0
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])
        self.eval_count += self.pop_size
        
        # Initialize velocity for PSO
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        # Track personal bests for PSO
        personal_best = np.copy(pop)
        personal_best_fitness = np.copy(pop_fitness)
        
        # Track global best
        global_best_idx = np.argmin(pop_fitness)
        global_best = np.copy(pop[global_best_idx])
        global_best_fitness = pop_fitness[global_best_idx]
        
        while self.eval_count < self.budget:
            # Differential Evolution step with adaptive F
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = pop[idxs]
                F = 0.5 + 0.5 * (1 - self.eval_count / self.budget)  # Adaptive mutation scaling
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness
                            
            # Particle Swarm Optimization step
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.w = 0.9 - ((0.9 - 0.4) * (self.eval_count / self.budget))  # Dynamically adjust inertia weight
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best[i] - pop[i]) +
                               self.c2 * r2 * (global_best - pop[i]))
                pop[i] = np.clip(pop[i] + velocity[i], lb, ub)
                
                current_fitness = func(pop[i])
                self.eval_count += 1
                if current_fitness < pop_fitness[i]:
                    pop_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[i]:
                        personal_best[i] = pop[i]
                        personal_best_fitness[i] = current_fitness
                        if current_fitness < global_best_fitness:
                            global_best = pop[i]
                            global_best_fitness = current_fitness
        
        return global_best, global_best_fitness