import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.learning_factor = 0.9
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 10
        self.num_swarms = 3  # Number of swarms for multi-swarm dynamics

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        temperature = self.initial_temperature
        population_size = self.initial_population_size

        # Initialize swarms
        swarms = []
        for _ in range(self.num_swarms):
            positions = np.random.rand(population_size, self.dim) * search_space + bounds[0]
            velocities = np.random.randn(population_size, self.dim) * 0.1 * search_space
            personal_best_positions = np.copy(positions)
            personal_best_scores = np.array([func(pos) for pos in positions])
            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            global_best_score = np.min(personal_best_scores)
            swarms.append((positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score))
        
        evaluations = population_size * self.num_swarms

        while evaluations < self.budget:
            for swarm in swarms:
                positions, velocities, personal_best_positions, personal_best_scores, global_best_position, global_best_score = swarm
                for i in range(population_size):
                    inertia = self.inertia_weight * velocities[i]
                    cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                    velocities[i] = inertia + cognitive_component + social_component
                    velocities[i] *= self.learning_factor
                    positions[i] += velocities[i]
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])
                    
                    if np.random.rand() < 0.1:  # Mutation strategy
                        positions[i] += np.random.randn(self.dim) * search_space * 0.05

                    score = func(positions[i])
                    evaluations += 1

                    if score < personal_best_scores[i]:
                        personal_best_positions[i] = positions[i]
                        personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = score

                    if np.random.rand() < np.exp(-abs(score - global_best_score) / temperature):
                        velocities[i] *= np.random.rand() * 2

                if evaluations / self.budget > 0.5:
                    population_size = max(self.min_population_size, population_size - 1)
                    positions = positions[:population_size]
                    velocities = velocities[:population_size]
                    personal_best_positions = personal_best_positions[:population_size]
                    personal_best_scores = personal_best_scores[:population_size]

                temperature *= self.cooling_rate

        best_swarm = min(swarms, key=lambda s: s[5])  # Find the best swarm based on global best score
        return best_swarm[4], best_swarm[5]