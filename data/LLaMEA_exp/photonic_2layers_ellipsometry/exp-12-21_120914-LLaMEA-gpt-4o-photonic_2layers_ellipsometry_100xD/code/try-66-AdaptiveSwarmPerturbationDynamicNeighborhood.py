import numpy as np

class AdaptiveSwarmPerturbationDynamicNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
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
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            
            # Define dynamic neighborhood radius
            neighborhood_radius = (ub - lb) * (1 - iter_count / self.budget)
            
            for i in range(self.population_size):
                distances = np.linalg.norm(swarm - swarm[i], axis=1)
                neighbors = np.where(distances < neighborhood_radius)[0]
                if len(neighbors) > 1:
                    local_best = personal_best[neighbors[np.argmin(personal_best_scores[neighbors])]]
                else:
                    local_best = personal_best[i]
                
                cognitive_component = np.random.uniform(size=self.dim) * (personal_best[i] - swarm[i])
                social_component = np.random.uniform(size=self.dim) * (local_best - swarm[i])
                
                velocities[i] = inertia_weight * velocities[i] + cognitive_component * np.random.rand() + social_component * np.random.rand()
                
            swarm += velocities
            swarm = np.clip(swarm, lb, ub)

            scores = np.array([func(ind) for ind in swarm])
            iter_count += self.population_size

            better_indices = scores < personal_best_scores
            personal_best[better_indices] = swarm[better_indices]
            personal_best_scores[better_indices] = scores[better_indices]

            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best = personal_best[min_index]
                global_best_score = personal_best_scores[min_index]

            if iter_count % (self.population_size * 5) == 0:
                perturbation = np.random.normal(0, 0.1 * (1 - iter_count / self.budget), size=(self.population_size, self.dim))
                swarm += perturbation
                self.history.append(global_best_score)
        
        return global_best_score, global_best