import numpy as np

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 2)  # Adjust population size based on budget
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.temperature = 100.0  # Initial temperature for annealing
        self.cooling_rate = 0.99  # Cooling rate for annealing

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]

        # Initialize positions and velocities
        positions = np.random.rand(self.population_size, self.dim) * search_space + bounds[0]
        velocities = np.random.randn(self.population_size, self.dim) * 0.1 * search_space
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity with dynamic inertia weight
                inertia = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component

                # Update position
                positions[i] += velocities[i]

                # Keep within bounds
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                # Evaluate new position
                score = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                if score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = score

                # Simulated annealing inspired exploration-exploitation balance
                if np.random.rand() < np.exp(-abs(score - global_best_score) / self.temperature):
                    velocities[i] *= np.random.rand() * 2

            # Cool down temperature
            self.temperature *= self.cooling_rate

        return global_best_position, global_best_score