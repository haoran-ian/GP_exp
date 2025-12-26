import numpy as np

class EnhancedSwarmPerturbationWithClustering:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.history = []
        self.memory = []

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

            cognitive_component = np.random.uniform(size=(self.population_size, self.dim)) * (personal_best - swarm)
            social_component = np.random.uniform(size=(self.population_size, self.dim)) * (global_best - swarm)
            
            velocities = inertia_weight * velocities + cognitive_component * np.random.rand() + social_component * np.random.rand()
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
                self.memory.append(swarm.copy())

                # Divisive Clustering
                cluster_centers = self.divisive_clustering(swarm)
                perturbation = np.random.normal(0, 0.1, size=(self.population_size, self.dim))
                for i in range(len(cluster_centers)):
                    swarm[i] = cluster_centers[i] + perturbation[i]
                
                self.history.append(global_best_score)

        return global_best_score, global_best

    def divisive_clustering(self, swarm):
        # A simple divisive clustering approach by bisecting the current swarm
        center = np.mean(swarm, axis=0)
        left_cluster = swarm[swarm[:, 0] < center[0]]
        right_cluster = swarm[swarm[:, 0] >= center[0]]
        cluster_centers = [np.mean(left_cluster, axis=0), np.mean(right_cluster, axis=0)]
        return cluster_centers