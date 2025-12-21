import numpy as np

class AdaptiveMemoryHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 * dim  # Slightly increased population size
        self.CR = 0.85  # Adjusted crossover probability
        self.F_min, self.F_max = 0.4, 0.8  # Adjusted mutation factors
        self.inertia_weight_max, self.inertia_weight_min = 0.85, 0.35  # Adjusted inertia weights
        self.cognitive = 1.6  # Adjusted cognitive coefficient
        self.social = 1.6  # Adjusted social coefficient
        self.memory_size = 15  # Increased memory size
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

    def adaptive_opposition_based_learning(self, population, lb, ub, generation):
        opp_population = lb + ub - population + 0.1 * np.random.uniform(-1, 1, population.shape) * generation / self.budget
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
            return self.memory_pos[np.random.randint(len(self.memory_pos))]  # Random selection for diversity
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
            generation = evaluations // (2 * self.population_size)
            opp_population1 = self.adaptive_opposition_based_learning(self.population1, lb, ub, generation)
            opp_population2 = self.adaptive_opposition_based_learning(self.population2, lb, ub, generation)
            for i in range(self.population_size):
                opp_score1 = func(opp_population1[i])
                opp_score2 = func(opp_population2[i])
                if opp_score1 < self.best_scores1[i]:
                    self.best_scores1[i] = opp_score1
                    self.best_positions1[i] = opp_population1[i]
                if opp_score2 < self.best_scores2[i]:
                    self.best_scores2[i] = opp_score2
                    self.best_positions2[i] = opp_population2[i]
                if opp_score1 < self.global_best_score:
                    self.global_best_score = opp_score1
                    self.global_best_position = opp_population1[i]
                if opp_score2 < self.global_best_score:
                    self.global_best_score = opp_score2
                    self.global_best_position = opp_population2[i]

            inertia_weight1 = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            inertia_weight2 = self.inertia_weight_min + (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                F1 = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                indices1 = [idx for idx in range(self.population_size) if idx != i]
                a1, b1, c1 = self.population1[np.random.choice(indices1, 3, replace=False)]
                mutant_vector1 = a1 + F1 * (b1 - c1)
                mutant_vector1 = np.clip(mutant_vector1, lb, ub)

                self.CR = 0.8 + 0.1 * (evaluations / self.budget)

                crossover_mask1 = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask1):
                    crossover_mask1[np.random.randint(0, self.dim)] = True
                trial_vector1 = np.where(crossover_mask1, mutant_vector1, self.population1[i])

                trial_score1 = func(trial_vector1)
                evaluations += 1

                if trial_score1 < self.best_scores1[i]:
                    self.best_scores1[i] = trial_score1
                    self.best_positions1[i] = trial_vector1

                if trial_score1 < self.global_best_score:
                    self.global_best_score = trial_score1
                    self.global_best_position = trial_vector1

                F2 = self.F_min + (self.F_max - self.F_min) * np.random.rand()
                indices2 = [idx for idx in range(self.population_size) if idx != i]
                a2, b2, c2 = self.population2[np.random.choice(indices2, 3, replace=False)]
                mutant_vector2 = a2 + F2 * (b2 - c2)
                mutant_vector2 = np.clip(mutant_vector2, lb, ub)

                crossover_mask2 = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask2):
                    crossover_mask2[np.random.randint(0, self.dim)] = True
                trial_vector2 = np.where(crossover_mask2, mutant_vector2, self.population2[i])

                trial_score2 = func(trial_vector2)
                evaluations += 1

                if trial_score2 < self.best_scores2[i]:
                    self.best_scores2[i] = trial_score2
                    self.best_positions2[i] = trial_vector2

                if trial_score2 < self.global_best_score:
                    self.global_best_score = trial_score2
                    self.global_best_position = trial_vector2

            self.population1, self.best_scores1 = self.stochastic_ranking(self.population1, self.best_scores1)
            self.population2, self.best_scores2 = self.stochastic_ranking(self.population2, self.best_scores2)

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component1 = self.cognitive * r1 * (self.best_positions1 - self.population1)
            social_component1 = self.social * r2 * (self.global_best_position - self.population1)
            self.velocities1 = (0.5 * (inertia_weight1 + inertia_weight2)) * self.velocities1 + cognitive_component1 + social_component1
            self.population1 = np.clip(self.population1 + self.velocities1, lb, ub)

            r3, r4 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive_component2 = self.cognitive * r3 * (self.best_positions2 - self.population2)
            social_component2 = self.social * r4 * (self.global_best_position - self.population2)
            self.velocities2 = (0.5 * (inertia_weight1 + inertia_weight2)) * self.velocities2 + cognitive_component2 + social_component2
            self.population2 = np.clip(self.population2 + self.velocities2, lb, ub)

            self.update_memory(self.global_best_position, self.global_best_score)

            memory_position = self.select_from_memory()
            if memory_position is not None:
                for i in range(self.population_size):
                    if np.random.rand() < 0.25:  # Further increased memory utilization rate
                        self.population1[i] = np.clip(memory_position + np.random.normal(0, 0.15, self.dim), lb, ub)  # Adjusted variance
                        self.population2[i] = np.clip(memory_position + np.random.normal(0, 0.15, self.dim), lb, ub)  # Adjusted variance

        return self.global_best_position, self.global_best_score