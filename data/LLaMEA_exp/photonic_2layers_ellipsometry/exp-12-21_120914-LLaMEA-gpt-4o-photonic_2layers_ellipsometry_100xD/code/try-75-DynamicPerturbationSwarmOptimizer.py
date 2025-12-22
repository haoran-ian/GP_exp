import numpy as np

class DynamicPerturbationSwarmOptimizer:
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
        velocity_scale = 0.5

        while iter_count < self.budget - self.population_size:
            inertia_weight = max(min_inertia, inertia_weight - 0.01)
            velocity_scale = max(0.1, velocity_scale - 0.005)  # Gradually reduce velocity scaling for exploitation

            cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_best - swarm)
            social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_best - swarm)
            
            velocities = inertia_weight * velocities + velocity_scale * (cognitive_component + social_component)
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

            # Dynamic population size adjustment
            if iter_count % (self.population_size * 5) == 0:
                self.population_size = min(self.population_size + 1, self.budget // 10)  # Updated line

            if iter_count % (self.population_size * 5) == 0:
                if len(self.history) > 2 and self.history[-1] >= self.history[-2]:
                    perturbation_stddev = max(0.05, 0.1 * (1.0 - (global_best_score - np.min(personal_best_scores)) / (np.max(personal_best_scores) - np.min(personal_best_scores) + 1e-10)))
                    perturbation = np.random.normal(0, perturbation_stddev, size=(self.population_size, self.dim))
                    swarm += perturbation
                self.history.append(global_best_score)

        return global_best_score, global_best