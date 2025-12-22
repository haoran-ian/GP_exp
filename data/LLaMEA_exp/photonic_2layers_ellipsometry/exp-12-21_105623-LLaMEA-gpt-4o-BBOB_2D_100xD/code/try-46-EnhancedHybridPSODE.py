import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.inertia = 0.5
        self.cognitive = 1.5
        self.social = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.velocity = np.random.rand(self.pop_size, dim)
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        personal_best = np.copy(swarm)
        personal_best_values = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            # Update inertia, cognitive, and social factors
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            self.cognitive = 1.5 + (0.5 * (eval_count / self.budget))
            self.social = 1.5 + (0.5 * (1 - eval_count / self.budget))
            self.velocity = self.inertia * self.velocity + self.cognitive * r1 * (personal_best - swarm) + self.social * r2 * (global_best - swarm)
            swarm = np.clip(swarm + self.velocity, lb, ub)
            
            new_population = np.copy(swarm)
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                # Differential Evolution Mutation
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = swarm[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                
                # Crossover
                trial = np.copy(swarm[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate or j == j_rand:
                        trial[j] = mutant[j]
                
                trial_value = func(trial)
                eval_count += 1
                
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value
                new_population[i] = trial if trial_value < func(swarm[i]) else swarm[i]
            swarm = new_population
        
        return global_best