import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Set population size
        self.w = 0.5       # Inertia weight for PSO
        self.c1 = 1.5      # Cognitive parameter for PSO
        self.c2 = 1.5      # Social parameter for PSO
        self.F = 0.8       # Mutation factor for DE
        self.CR = 0.9      # Crossover rate for DE
        
    def __call__(self, func):
        # Initialize swarm
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        eval_count = self.pop_size
        
        while eval_count < self.budget:
            # PSO Update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - swarm) +
                          self.c2 * r2 * (global_best_position - swarm))
            swarm = np.clip(swarm + velocities, lower_bound, upper_bound)
            
            # DE Update
            for i in range(self.pop_size):
                # Select three random indices different from i
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation operation
                mutant = np.clip(swarm[a] + self.F * (swarm[b] - swarm[c]), lower_bound, upper_bound)
                
                # Crossover operation
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):  # Ensure at least one dimension crosses over
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, swarm[i])
                
                # Evaluate new candidate
                trial_score = func(trial)
                eval_count += 1
                
                # Selection operation
                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    
                    # Update global best
                    if trial_score < func(global_best_position):
                        global_best_position = trial
            
            # Break if evaluation budget is exceeded
            if eval_count >= self.budget:
                break
        
        return global_best_position