import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.inertia = 0.9  # Start with higher inertia for exploration
        self.cognitive = 2.0
        self.social = 2.0
        self.elitism_rate = 0.1
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
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            # Adaptive inertia decreases over time to favor exploitation
            self.inertia = 0.9 - (0.5 * (eval_count / self.budget))
            self.cognitive = 2.0 - (0.5 * (eval_count / self.budget))
            self.social = 2.0 + (0.5 * (eval_count / self.budget))
            self.velocity = self.inertia * self.velocity + self.cognitive * r1 * (personal_best - swarm) + self.social * r2 * (global_best - swarm)
            swarm = np.clip(swarm + self.velocity, lb, ub)
            
            new_personal_best = np.copy(personal_best)
            new_personal_best_values = np.copy(personal_best_values)
            
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break
                trial = np.copy(swarm[i])
                if np.random.rand() < self.crossover_rate:
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() > self.crossover_rate and j != j_rand:
                            trial[j] = swarm[np.random.randint(self.pop_size)][j]
                trial_value = func(trial)
                eval_count += 1
                
                if trial_value < new_personal_best_values[i]:
                    new_personal_best[i] = trial
                    new_personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value
            
            # Elitism: keep a portion of the best solutions
            sorted_indices = np.argsort(new_personal_best_values)
            elite_count = int(self.elitism_rate * self.pop_size)
            for i in range(elite_count):
                idx = sorted_indices[i]
                personal_best[idx] = new_personal_best[idx]
                personal_best_values[idx] = new_personal_best_values[idx]
        
        return global_best