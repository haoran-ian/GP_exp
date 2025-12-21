import numpy as np

class EnhancedAdaptiveMemoryHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.CR = 0.9
        self.F_min, self.F_max = 0.4, 0.8
        self.inertia_weight_max, self.inertia_weight_min = 0.8, 0.3
        self.cognitive = 1.7
        self.social = 1.7
        self.memory_size = 5
        self.memory_pos = []
        self.memory_scores = []
        self.population = None
        self.velocities = None
        self.best_positions = None
        self.best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf

    def adaptive_opposition_based_learning(self, population, lb, ub, generation):
        shrink_factor = 1 - (generation / self.budget)
        opp_population = lb + ub - population + 0.2 * np.random.uniform(-1, 1, population.shape) * shrink_factor
        opp_population = np.clip(opp_population, lb, ub)
        return opp_population

    def stochastic_ranking(self, population, scores, probability=0.45):
        indices = np.arange(len(scores))
        for i in range(len(scores)):
            for j in range(len(scores) - 1):
                if (scores[indices[j]] > scores[indices[j + 1]]) or (np.random.rand() < probability):
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
        return population[indices], scores[indices]

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
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, np.inf)
        evaluations = 0

        while evaluations < self.budget:
            generation = evaluations // self.population_size
            opp_population = self.adaptive_opposition_based_learning(self.population, lb, ub, generation)

            for i in range(self.population_size):
                opp_score = func(opp_population[i])
                if opp_score < self.best_scores[i]:
                    self.best_scores[i] = opp_score
                    self.best_positions[i] = opp_population[i]
                if opp_score < self.global_best_score:
                    self.global_best_score = opp_score
                    self.global_best_position = opp_population[i]

            inertia_weight = (self.inertia_weight_max - self.inertia_weight_min) * (1 - evaluations / self.budget) + self.inertia_weight_min

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                F = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + F * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                self.CR = 0.8 + 0.1 * np.random.rand()

                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.best_scores[i]:
                    self.best_scores[i] = trial_score
                    self.best_positions[i] = trial_vector

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

            self.population, self.best_scores = self.stochastic_ranking(self.population, self.best_scores)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive * r1 * (self.best_positions - self.population)
            social_component = self.social * r2 * (self.global_best_position - self.population)
            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component
            self.population = np.clip(self.population + self.velocities, lb, ub)

            self.update_memory(self.global_best_position, self.global_best_score)

            memory_position = self.select_from_memory()
            if memory_position is not None:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:
                        self.population[i] = np.clip(memory_position + np.random.normal(0, 0.1, self.dim), lb, ub)

        return self.global_best_position, self.global_best_score