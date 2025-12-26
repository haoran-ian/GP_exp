import numpy as np

class IterativeHierarchicalSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 20
        self.layers = 3
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize swarms for hierarchical layers
        swarms = [np.random.uniform(lb, ub, (self.base_population_size, self.dim)) for _ in range(self.layers)]
        velocities = [np.zeros((self.base_population_size, self.dim)) for _ in range(self.layers)]
        personal_best = [swarm.copy() for swarm in swarms]
        personal_best_scores = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        
        # Identify global best across all layers
        all_bests = np.array([p[np.argmin(s)] for p, s in zip(personal_best, personal_best_scores)])
        global_best_index = np.argmin([func(ind) for ind in all_bests])
        global_best = all_bests[global_best_index]
        global_best_score = func(global_best)
        self.history.append(global_best_score)

        iter_count = 0
        inertia_weight = 0.9
        min_inertia = 0.4

        while iter_count < self.budget - self.base_population_size * self.layers:
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            for i in range(self.layers):
                cognitive_component = np.random.uniform(size=(self.base_population_size, self.dim)) * (personal_best[i] - swarms[i])
                social_component = np.random.uniform(size=(self.base_population_size, self.dim)) * (global_best - swarms[i])
                
                velocities[i] = inertia_weight * velocities[i] + cognitive_component * np.random.rand() + social_component * np.random.rand()
                swarms[i] += velocities[i]

                swarms[i] = np.clip(swarms[i], lb, ub)

                scores = np.array([func(ind) for ind in swarms[i]])
                iter_count += self.base_population_size

                better_indices = scores < personal_best_scores[i]
                personal_best[i][better_indices] = swarms[i][better_indices]
                personal_best_scores[i][better_indices] = scores[better_indices]

            # Update global best
            all_bests = np.array([p[np.argmin(s)] for p, s in zip(personal_best, personal_best_scores)])
            global_best_index = np.argmin([func(ind) for ind in all_bests])
            if func(all_bests[global_best_index]) < global_best_score:
                global_best = all_bests[global_best_index]
                global_best_score = func(global_best)

            # Introduce fitness-sharing for diversity enhancement every few iterations
            if iter_count % (self.base_population_size * self.layers * 5) == 0:
                for i in range(self.layers):
                    if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                        perturbation = np.random.normal(0, 0.1 / (i + 1), size=(self.base_population_size, self.dim))
                        swarms[i] += perturbation
                    self.history.append(global_best_score)
        
        return global_best_score, global_best