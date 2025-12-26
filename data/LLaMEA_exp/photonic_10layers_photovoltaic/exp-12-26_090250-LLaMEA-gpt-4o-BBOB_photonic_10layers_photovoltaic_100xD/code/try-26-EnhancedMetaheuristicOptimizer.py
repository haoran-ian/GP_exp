import numpy as np

class EnhancedMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.learning_factor = 0.9
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 10
        self.population_fractions = [0.6, 0.4]  # Use two subpopulations

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        temperature = self.initial_temperature
        population_size = self.initial_population_size
        evaluations = 0

        # Initialize subpopulations
        subpopulations = [int(population_size * f) for f in self.population_fractions]
        positions = [np.random.rand(s, self.dim) * search_space + bounds[0] for s in subpopulations]
        velocities = [np.random.randn(s, self.dim) * 0.1 * search_space for s in subpopulations]
        personal_best_positions = [np.copy(p) for p in positions]
        personal_best_scores = [np.array([func(pos) for pos in p]) for p in positions]

        global_best_position = np.array(min((p[np.argmin(s)] for p, s in zip(personal_best_positions, personal_best_scores)), key=func))
        global_best_score = func(global_best_position)

        evaluations += sum(subpopulations)

        while evaluations < self.budget:
            for k in range(len(positions)):
                for i in range(subpopulations[k]):
                    # Update velocity with adaptive inertia
                    velocity_inertia = self.inertia_weight * (1 - evaluations / self.budget) * velocities[k][i]
                    cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[k][i] - positions[k][i])
                    social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[k][i])
                    velocities[k][i] = velocity_inertia + cognitive_component + social_component
                    velocities[k][i] *= self.learning_factor

                    # Update position
                    positions[k][i] += velocities[k][i]
                    positions[k][i] = np.clip(positions[k][i], bounds[0], bounds[1])

                    # Evaluate new position
                    score = func(positions[k][i])
                    evaluations += 1

                    # Update personal and global bests
                    if score < personal_best_scores[k][i]:
                        personal_best_positions[k][i] = positions[k][i]
                        personal_best_scores[k][i] = score

                    if score < global_best_score:
                        global_best_position = positions[k][i]
                        global_best_score = score

                    # Simulated annealing inspired exploration-exploitation balance
                    if np.random.rand() < np.exp(-abs(score - global_best_score) / temperature):
                        velocities[k][i] *= np.random.rand() * 2

            # Adaptive population size reduction
            if evaluations / self.budget > 0.5:
                for k in range(len(subpopulations)):
                    subpopulations[k] = max(self.min_population_size, subpopulations[k] - 1)
                    positions[k] = positions[k][:subpopulations[k]]
                    velocities[k] = velocities[k][:subpopulations[k]]
                    personal_best_positions[k] = personal_best_positions[k][:subpopulations[k]]
                    personal_best_scores[k] = personal_best_scores[k][:subpopulations[k]]

            # Cool down temperature gradually
            temperature *= self.cooling_rate

        return global_best_position, global_best_score