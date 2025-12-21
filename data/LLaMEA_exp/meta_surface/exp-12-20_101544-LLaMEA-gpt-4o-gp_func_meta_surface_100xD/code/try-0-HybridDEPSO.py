import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # You can adjust this parameter
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.inertia_weight = 0.5  # Inertia weight for PSO
        self.cognitive = 1.5      # Cognitive coefficient for PSO
        self.social = 1.5         # Social coefficient for PSO
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + self.F * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                # Selection
                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.best_scores[i]:
                    self.best_scores[i] = trial_score
                    self.best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

            # Particle Swarm Optimization Update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive * r1 * (self.best_positions - self.population)
            social_component = self.social * r2 * (self.global_best_position - self.population)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, lb, ub)

        return self.global_best_position, self.global_best_score