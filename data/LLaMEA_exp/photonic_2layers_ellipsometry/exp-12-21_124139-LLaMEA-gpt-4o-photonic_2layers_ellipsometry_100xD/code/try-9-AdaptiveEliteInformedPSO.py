import numpy as np

class AdaptiveEliteInformedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_scale = 0.1
        self.inertia_weight_decay = (0.7 - 0.3) / self.budget

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize swarm positions and velocities
        positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(p) for p in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        # Initialize elite solutions
        elite_count = max(2, int(self.population_size * 0.1))
        elites = positions[np.argsort(personal_best_scores)[:elite_count]]

        while self.evaluations < self.budget:
            previous_global_best_position = np.copy(global_best_position)

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity with elite influence
                elite_influence = np.mean(elites, axis=0)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                    + self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                    + self.mutation_scale * np.random.rand(self.dim) * (elite_influence - positions[i])
                )

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)

                # Evaluate and update personal bests
                score = func(positions[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_position = positions[i]
                        global_best_score = score
            
            # Update elite solutions
            elites = positions[np.argsort(personal_best_scores)[:elite_count]]
            
            # Adaptive adjustment of inertia weight and coefficients
            self.inertia_weight = max(0.3, self.inertia_weight - self.inertia_weight_decay)
            score_diff = (global_best_score / (np.mean(personal_best_scores) + 1e-8))
            self.cognitive_coeff = min(2.0, self.cognitive_coeff + score_diff * 0.1)
            self.social_coeff = min(2.0, self.social_coeff + score_diff * 0.1)

            # Ensure elitism
            if func(previous_global_best_position) < global_best_score:
                global_best_position = previous_global_best_position

        return global_best_position