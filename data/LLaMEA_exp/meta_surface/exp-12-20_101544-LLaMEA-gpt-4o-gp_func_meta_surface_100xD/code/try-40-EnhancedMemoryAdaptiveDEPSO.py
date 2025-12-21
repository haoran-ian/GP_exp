import numpy as np

class EnhancedMemoryAdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.min_CR, self.max_CR = 0.7, 0.9
        self.min_F, self.max_F = 0.4, 0.9
        self.inertia_weight_max, self.inertia_weight_min = 0.9, 0.4
        self.cognitive = 2.0
        self.social = 2.0
        self.memory_size = 5
        self.memory_pos = []
        self.memory_scores = []
        self.population1 = None
        self.population2 = None
        self.velocities1 = None
        self.velocities2 = None
        self.best_positions1 = None
        self.best_positions2 = None
        self.best_scores1 = None
        self.best_scores2 = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def adaptive_differential_mutation(self, population, lb, ub, i):
        F = self.min_F + (self.max_F - self.min_F) * (1 - (i / self.budget) ** 2)
        indices = [idx for idx in range(self.population_size) if idx != i]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant_vector = np.clip(a + F * (b - c), lb, ub)
        return mutant_vector

    def update_memory(self, position, score):
        if len(self.memory_scores) < self.memory_size:
            self.memory_pos.append(position)
            self.memory_scores.append(score)
        else:
            worst_idx = np.argmax(self.memory_scores)
            if score < self.memory_scores[worst_idx]:
                self.memory_pos[worst_idx] = position
                self.memory_scores[worst_idx] = score

    def select_from_memory(self):
        if self.memory_scores:
            return self.memory_pos[np.argmin(self.memory_scores)]
        return None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.population1 = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.population2 = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities1 = np.zeros((self.population_size, self.dim))
        self.velocities2 = np.zeros((self.population_size, self.dim))
        self.best_positions1 = np.copy(self.population1)
        self.best_positions2 = np.copy(self.population2)
        self.best_scores1 = np.full(self.population_size, np.inf)
        self.best_scores2 = np.full(self.population_size, np.inf)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                mutant_vector1 = self.adaptive_differential_mutation(self.population1, lb, ub, evaluations)
                mutant_vector2 = self.adaptive_differential_mutation(self.population2, lb, ub, evaluations)

                CR = self.min_CR + (self.max_CR - self.min_CR) * (evaluations / self.budget)
                crossover_mask1 = np.random.rand(self.dim) < CR
                if not np.any(crossover_mask1):
                    crossover_mask1[np.random.randint(0, self.dim)] = True
                trial_vector1 = np.where(crossover_mask1, mutant_vector1, self.population1[i])

                crossover_mask2 = np.random.rand(self.dim) < CR
                if not np.any(crossover_mask2):
                    crossover_mask2[np.random.randint(0, self.dim)] = True
                trial_vector2 = np.where(crossover_mask2, mutant_vector2, self.population2[i])

                trial_score1 = func(trial_vector1)
                evaluations += 1
                trial_score2 = func(trial_vector2)
                evaluations += 1

                if trial_score1 < self.best_scores1[i]:
                    self.best_scores1[i] = trial_score1
                    self.best_positions1[i] = trial_vector1

                if trial_score2 < self.best_scores2[i]:
                    self.best_scores2[i] = trial_score2
                    self.best_positions2[i] = trial_vector2

                if trial_score1 < self.global_best_score:
                    self.global_best_score = trial_score1
                    self.global_best_position = trial_vector1

                if trial_score2 < self.global_best_score:
                    self.global_best_score = trial_score2
                    self.global_best_position = trial_vector2

            inertia_weight1 = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            inertia_weight2 = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component1 = self.cognitive * r1 * (self.best_positions1 - self.population1)
            social_component1 = self.social * r2 * (self.global_best_position - self.population1)
            self.velocities1 = (inertia_weight1) * self.velocities1 + cognitive_component1 + social_component1
            self.population1 = np.clip(self.population1 + self.velocities1, lb, ub)

            r3, r4 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component2 = self.cognitive * r3 * (self.best_positions2 - self.population2)
            social_component2 = self.social * r4 * (self.global_best_position - self.population2)
            self.velocities2 = (inertia_weight2) * self.velocities2 + cognitive_component2 + social_component2
            self.population2 = np.clip(self.population2 + self.velocities2, lb, ub)

            self.update_memory(self.global_best_position, self.global_best_score)

            memory_position = self.select_from_memory()
            if memory_position is not None:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        self.population1[i] = np.clip(memory_position + np.random.normal(0, 0.1, self.dim), lb, ub)
                        self.population2[i] = np.clip(memory_position + np.random.normal(0, 0.1, self.dim), lb, ub)

        return self.global_best_position, self.global_best_score