import numpy as np

class EnhancedAdaptiveHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + int(2 * np.sqrt(dim))
        self.inertia = 0.9
        self.min_inertia = 0.4
        self.cognitive = 1.5
        self.social = 1.5
        self.mutation_factor = 0.8
        self.min_mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.velocity = np.random.rand(self.pop_size, dim)
        self.perturbation_rate = 0.1  # Additional perturbation rate for diversity

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
            
            # Self-adaptive adjustment of inertia and mutation factor
            dynamic_factor = np.exp(-eval_count / (self.budget / 2))
            self.inertia = self.min_inertia + (self.inertia - self.min_inertia) * dynamic_factor
            self.mutation_factor = self.min_mutation_factor + (self.mutation_factor - self.min_mutation_factor) * dynamic_factor
            
            # Velocity and position update with perturbation
            self.velocity = (self.inertia * self.velocity 
                             + self.cognitive * r1 * (personal_best - swarm) 
                             + self.social * r2 * (global_best - swarm))
            swarm = np.clip(swarm + self.velocity, lb, ub)
            
            # Introduce random perturbation to enhance exploration
            if np.random.rand() < self.perturbation_rate:
                perturbation = np.random.normal(scale=(ub - lb) / 10, size=(self.pop_size, self.dim))
                swarm = np.clip(swarm + perturbation, lb, ub)
            
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