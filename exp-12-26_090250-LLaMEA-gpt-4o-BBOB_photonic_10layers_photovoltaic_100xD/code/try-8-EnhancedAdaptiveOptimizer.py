import numpy as np

class EnhancedAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.learning_factor = 0.9
        self.initial_temperature = 100.0
        self.cooling_rate = 0.995
        self.min_population_size = 10
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        temperature = self.initial_temperature
        population_size = self.initial_population_size
        
        positions = np.random.rand(population_size, self.dim) * search_space + bounds[0]
        velocities = np.random.randn(population_size, self.dim) * 0.1 * search_space
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = positions[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), bounds[0], bounds[1])
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(cross_points, mutant, positions[i])
                
                score = func(trial)
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = score
                
                if score < global_best_score:
                    global_best_position = trial
                    global_best_score = score

                # Particle Swarm Optimization inspired update
                inertia = self.inertia_weight * velocities[i]
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (personal_best_positions[i] - positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia + cognitive_component + social_component
                velocities[i] *= self.learning_factor
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], bounds[0], bounds[1])

            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = positions[:population_size]
                velocities = velocities[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]

            temperature *= self.cooling_rate

        return global_best_position, global_best_score