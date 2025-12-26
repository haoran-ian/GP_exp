import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.inertia = self.inertia_max
        self.cognitive = 2.0
        self.social = 2.0
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
            self.inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (eval_count / self.budget))
            self.velocity = (self.inertia * self.velocity + 
                             self.cognitive * r1 * (personal_best - swarm) +
                             self.social * r2 * (global_best - swarm))
            swarm = np.clip(swarm + self.velocity, lb, ub)
            
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
                
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value
                
                # Local search enhancement
                if np.random.rand() < 0.1:  # Small chance of local exploration
                    local_trial = np.clip(trial + np.random.normal(0, 0.1, self.dim), lb, ub)
                    local_trial_value = func(local_trial)
                    eval_count += 1
                    
                    if local_trial_value < trial_value:
                        personal_best[i] = local_trial
                        personal_best_values[i] = local_trial_value
                        if local_trial_value < global_best_value:
                            global_best = local_trial
                            global_best_value = local_trial_value
        
        return global_best