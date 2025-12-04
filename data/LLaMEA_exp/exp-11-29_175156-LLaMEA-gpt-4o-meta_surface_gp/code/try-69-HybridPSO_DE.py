import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9  # Changed initial inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.F = 0.5  # Differential Evolution mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.full(self.population_size, float('inf'))
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = float('inf')

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the population
            for i in range(self.population_size):
                value = func(self.population[i])
                evaluations += 1
                if value < self.personal_best_value[i]:
                    self.personal_best_value[i] = value
                    self.personal_best_position[i] = self.population[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.population[i]

            # Update velocity and position based on PSO
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.population_size):
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    (self.cognitive_coeff + 0.05 * np.sin(evaluations)) * r1 * (self.personal_best_position[i] - self.population[i]) +
                                    self.social_coeff * r2 * (self.global_best_position - self.population[i])) * (1 + 0.1 * np.sin(evaluations))
                self.population[i] = np.clip(self.population[i] + self.velocity[i] * (1 + 0.05 * np.cos(evaluations)), self.lower_bound, self.upper_bound)

            # Adaptively reduce inertia weight
            self.inertia_weight = max(0.3, self.inertia_weight * (0.95 + 0.03 * np.sin(evaluations * 3) + 0.02 * np.cos(evaluations * 2)))  # Modified inertia weight with an additional cosine component

            # Apply DE mutation strategy with adaptive F
            self.F = 0.6 + 0.4 * np.cos(evaluations * np.pi / self.budget) + 0.05 * np.sin(evaluations)  # Adaptive F factor with sinusoidal enhancement
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                trial_value = func(trial)
                evaluations += 1
                if trial_value < self.personal_best_value[i]:
                    self.population[i] = trial
                    self.personal_best_value[i] = trial_value
                    self.personal_best_position[i] = trial
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial

            # Update social coefficient for faster convergence
            self.social_coeff = 1.5 + 0.1 * np.cos(evaluations * np.pi / self.budget)  # Dynamic adjustment using cosine
            # Adaptively adjust crossover probability
            self.CR = 0.9 - 0.4 * evaluations / self.budget  # Add adaptive crossover probability adjustment
            
            # Dynamic population size adjustment based on fitness diversity
            if evaluations % 50 == 0:  # Check every 50 evaluations
                diversity = np.std(self.personal_best_value)
                if diversity < 1e-3:  # If diversity is low, increase population size for exploration
                    self.population_size = min(self.population_size + 5, 30)
                else:  # If diversity is sufficient, reduce for exploitation
                    self.population_size = max(self.population_size - 5, 15)

        return self.global_best_position, self.global_best_value