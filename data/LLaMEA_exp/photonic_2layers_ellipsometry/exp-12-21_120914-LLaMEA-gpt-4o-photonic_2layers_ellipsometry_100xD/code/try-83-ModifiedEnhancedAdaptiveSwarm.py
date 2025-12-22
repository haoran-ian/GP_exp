import numpy as np

class ModifiedEnhancedAdaptiveSwarm:
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
            
            # Adaptive learning factors
            cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_best - swarm)
            social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_best - swarm)
            
            cognitive_factor = 2.5 - (2.0 * iter_count) / self.budget  # New line
            social_factor = 0.5 + (2.0 * iter_count) / self.budget  # New line
            
            velocities = inertia_weight * velocities + \
                         cognitive_component * cognitive_factor + \
                         social_component * social_factor  # Modified line
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
                self.population_size = min(self.population_size + 1, self.budget // 10)

            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                    swarm += perturbation
                self.history.append(global_best_score)

            # Local search operation
            if iter_count % (self.population_size * 10) == 0:  # New line
                local_search_idx = np.random.randint(self.population_size)  # New line
                local_search_ind = swarm[local_search_idx] + np.random.uniform(-0.05, 0.05, self.dim)  # New line
                local_search_score = func(np.clip(local_search_ind, lb, ub))  # New line
                if local_search_score < personal_best_scores[local_search_idx]:  # New line
                    personal_best[local_search_idx] = local_search_ind  # New line
                    personal_best_scores[local_search_idx] = local_search_score  # New line
        
        return global_best_score, global_best