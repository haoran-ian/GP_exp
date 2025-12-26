import numpy as np

class EnhancedAdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.inertia = 0.9  # Start with a higher inertia for exploration
        self.min_inertia = 0.4  # Minimum inertia for exploitation
        self.cognitive = 1.5
        self.social = 1.5
        self.mutation_factor = 0.8
        self.min_mutation_factor = 0.5
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
        diversity_threshold = np.var(swarm) * 0.1  # Set diversity threshold

        while eval_count < self.budget:
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            
            # Dynamic adjustment of inertia and mutation factor
            self.inertia = self.min_inertia + ((self.inertia - self.min_inertia) * (self.budget - eval_count) / self.budget)
            self.mutation_factor = self.min_mutation_factor + ((self.mutation_factor - self.min_mutation_factor) * (self.budget - eval_count) / self.budget)
            
            self.cognitive = 1.5 + (0.5 * (eval_count / self.budget))
            self.social = 1.5 + (0.5 * (1 - eval_count / self.budget))
            self.velocity = self.inertia * self.velocity + self.cognitive * r1 * (personal_best - swarm) + self.social * r2 * (global_best - swarm)
            swarm = np.clip(swarm + self.velocity, lb, ub)
            
            # Diversity preservation strategy
            if np.var(swarm) < diversity_threshold:
                indices_to_mutate = np.random.choice(self.pop_size, size=self.pop_size // 2, replace=False)
                mutation_noise = np.random.normal(0, 0.1, (len(indices_to_mutate), self.dim))
                swarm[indices_to_mutate] += mutation_noise
                swarm = np.clip(swarm, lb, ub)
            
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
        
        return global_best