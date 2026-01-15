import numpy as np
from sklearn.cluster import MiniBatchKMeans

class SAGS_Enhanced_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.1  # Initial learning rate
        self.beta = 0.9   # Momentum factor
        self.mutation_rate = 0.1
        self.population_size = 10
        self.best_position = None
        self.best_value = float('inf')
        self.inertia_weight = 0.9  # Inertia weight for velocity update

    def __call__(self, func):
        # Initialize swarm positions and velocities
        positions = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        
        # Evaluate initial positions
        values = np.array([func(pos) for pos in positions])
        evaluations = self.population_size
        
        # Identify initial best position
        best_idx = np.argmin(values)
        self.best_value = values[best_idx]
        self.best_position = positions[best_idx].copy()

        # Dynamic learning rate, mutation, and inertia schedule
        alpha_schedule = lambda evals: self.alpha * (1 - evals / self.budget)
        mutation_schedule = lambda evals: self.mutation_rate * (1 - evals / self.budget)
        inertia_schedule = lambda evals: self.inertia_weight * (1 - evals / (2 * self.budget))
        
        previous_best_value = self.best_value

        while evaluations < self.budget:
            # Adaptive inertia weight for velocity update
            inertia_weight = inertia_schedule(evaluations)

            for i in range(self.population_size):
                gradient = np.random.normal(scale=0.1, size=self.dim)
                adaptive_alpha = alpha_schedule(evaluations)
                velocities[i] = (inertia_weight * velocities[i] - adaptive_alpha * gradient 
                                 + 0.2 * (self.best_position - positions[i]))
            
            # Update positions
            positions = positions + velocities
            positions = np.clip(positions, func.bounds.lb, func.bounds.ub)
            
            # Evaluate new positions
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                value = func(positions[i])
                evaluations += 1
                
                if value < values[i]:
                    values[i] = value
                    if value < self.best_value:
                        self.best_value = value
                        self.best_position = positions[i].copy()

            # Clustering for diversity exploitation
            if evaluations < self.budget:
                kmeans = MiniBatchKMeans(n_clusters=max(2, self.population_size // 2))
                clusters = kmeans.fit_predict(positions)
                
                for cluster_id in set(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    if len(cluster_indices) > 1:
                        cluster_pos = positions[cluster_indices]
                        cluster_vals = values[cluster_indices]
                        cluster_best_idx = np.argmin(cluster_vals)
                        cluster_best_pos = cluster_pos[cluster_best_idx]
                        
                        for i in cluster_indices:
                            if evaluations >= self.budget:
                                break
                            offspring = cluster_best_pos + np.random.normal(0, mutation_schedule(evaluations), self.dim)
                            offspring = np.clip(offspring, func.bounds.lb, func.bounds.ub)
                            value_offspring = func(offspring)
                            evaluations += 1
                            
                            if value_offspring < values[i]:
                                positions[i] = offspring
                                values[i] = value_offspring
                                if value_offspring < self.best_value:
                                    self.best_value = value_offspring
                                    self.best_position = offspring.copy()
            
            previous_best_value = self.best_value

        return self.best_position, self.best_value