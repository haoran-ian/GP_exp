import numpy as np

class EnhancedAdaptiveSwarmPerturbation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initial swarm setup
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_scores = np.array([func(ind) for ind in swarm])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        self.history.append(global_best_score)
        
        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4

        while iter_count < self.budget - self.population_size:
            # Update inertia weight dynamically
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            # Update velocities and positions
            cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_best - swarm)
            social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_best - swarm)
            velocities = inertia_weight * velocities + cognitive_component + social_component
            swarm += velocities
            
            # Clip positions to bounds
            swarm = np.clip(swarm, lb, ub)
            
            # Evaluate new solutions
            scores = np.array([func(ind) for ind in swarm])
            iter_count += self.population_size
            
            # Update personal bests
            better_indices = scores < personal_best_scores
            personal_best[better_indices] = swarm[better_indices]
            personal_best_scores[better_indices] = scores[better_indices]
            
            # Update global best
            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best = personal_best[min_index]
                global_best_score = personal_best_scores[min_index]
                
            # Nonlinear adaptive perturbation based on success history
            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                    swarm += perturbation
                self.history.append(global_best_score)
        
            # Dynamic population adjustment
            if global_best_score < np.mean(personal_best_scores):
                self.population_size = min(self.population_size + 1, 50)
        
        return global_best_score, global_best