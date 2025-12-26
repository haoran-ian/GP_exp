import numpy as np

class AdaptiveEvolutionaryStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 5)
        self.min_population_size = 10
        self.temperature = 100.0
        self.cooling_rate = 0.995
        self.learning_rate = 0.1
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        search_space = bounds[1] - bounds[0]
        population_size = self.initial_population_size
        
        positions = np.random.rand(population_size, self.dim) * search_space + bounds[0]
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size

        while evaluations < self.budget:
            # Adaptive mutation and crossover
            mutation_prob = np.clip(1.0 - (evaluations / self.budget), 0.1, 0.9)
            crossover_prob = np.clip(evaluations / self.budget, 0.1, 0.9)

            for i in range(population_size):
                # Mutation
                if np.random.rand() < mutation_prob:
                    mutation = self.learning_rate * np.random.randn(self.dim)
                    positions[i] += mutation
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                # Crossover (Blend crossover)
                if np.random.rand() < crossover_prob:
                    partner_index = np.random.choice(population_size)
                    alpha = np.random.rand(self.dim)
                    positions[i] = alpha * positions[i] + (1 - alpha) * positions[partner_index]
                    positions[i] = np.clip(positions[i], bounds[0], bounds[1])

                score = func(positions[i])
                evaluations += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_scores[i] = score

                if score < global_best_score:
                    global_best_position = positions[i]
                    global_best_score = score

            # Cool down the system
            self.temperature *= self.cooling_rate

            # Adapt population size
            if evaluations / self.budget > 0.5:
                population_size = max(self.min_population_size, population_size - 1)
                positions = positions[:population_size]
                personal_best_positions = personal_best_positions[:population_size]
                personal_best_scores = personal_best_scores[:population_size]

        return global_best_position, global_best_score