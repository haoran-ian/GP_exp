import numpy as np

class ACPO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.crossover_probability = 0.7

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate the population
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                score = func(population[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i] * (1 - 0.1 * np.sin(0.05 * evaluations))  # Dynamic personal best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]
            
            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                social_velocity = self.social_coefficient * r2 * (global_best_position - population[i])
                self.inertia_weight = 0.5 + (0.5 * np.sin(evaluations)) * 0.98  # Modified damping factor
                self.cognitive_coefficient = 1.5 + 0.5 * np.sin(0.05 * evaluations) + 0.1  # Adjusted cognitive coefficient
                adaptive_learning_factor = 1.0 + 0.4 * np.sin(0.1 * evaluations)  # New adaptive learning factor
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity) * adaptive_learning_factor
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
            
            # Perform adaptive crossover
            for i in range(0, self.population_size, 2):
                if evaluations >= self.budget:
                    break
                self.crossover_probability = 0.65 + 0.15 * np.cos(evaluations)  # Adaptive crossover probability
                if np.random.rand() < self.crossover_probability:
                    parent1, parent2 = population[i], population[(i+1) % self.population_size]
                    mask = np.random.rand(self.dim) < np.sin(0.05 * evaluations)  # Dynamic crossover mask
                    offspring1 = np.where(mask, parent1, parent2)
                    offspring2 = np.where(mask, (parent1 + parent2) / 2, parent1)  # Modified offspring creation
                    population[i], population[(i+1) % self.population_size] = offspring1, offspring2

        return global_best_position, global_best_score