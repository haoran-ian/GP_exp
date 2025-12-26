import numpy as np

class AHSO_Adaptive_MultiSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1_init = 2.5
        self.c2_init = 0.5
        self.inertia_weight = 0.729
        self.mutation_prob = 0.2
        self.crossover_prob = 0.7
        self.elite_fraction = 0.1
        self.alpha = 1.5
        self.num_swarms = 5
        self.F = 0.8  # Differential evolution factor

    def levy_flight(self, size):
        sigma = (np.math.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v) ** (1 / self.alpha)
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        populations = [np.random.uniform(lb, ub, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(pop) for pop in populations]
        personal_best_scores = [np.full(self.population_size, np.inf) for _ in range(self.num_swarms)]
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            for k in range(self.num_swarms):
                scores = np.array([func(ind) for ind in populations[k]])
                evaluations += len(populations[k])

                better_positions = scores < personal_best_scores[k]
                personal_best_positions[k][better_positions] = populations[k][better_positions]
                personal_best_scores[k][better_positions] = scores[better_positions]

                min_score_idx = np.argmin(personal_best_scores[k])
                if personal_best_scores[k][min_score_idx] < global_best_score:
                    global_best_position = personal_best_positions[k][min_score_idx]
                    global_best_score = personal_best_scores[k][min_score_idx]

                # Adaptive learning factors
                self.c1 = self.c1_init * (1 - evaluations / self.budget)
                self.c2 = self.c2_init * (evaluations / self.budget)

                self.inertia_weight = 0.729 * (0.4 ** (evaluations / self.budget))
                r1, r2 = np.random.rand(2, self.population_size, self.dim)
                velocities[k] = (self.inertia_weight * velocities[k] +
                                 self.c1 * r1 * (personal_best_positions[k] - populations[k]) +
                                 self.c2 * r2 * (global_best_position - populations[k]))
                velocities[k] = np.clip(velocities[k], -0.2 * (ub - lb), 0.2 * (ub - lb))
                populations[k] += velocities[k]
                populations[k] = np.clip(populations[k], lb, ub)

                # Differential evolution step
                for i in range(self.population_size):
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = populations[k][indices]
                    mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, populations[k][i])
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < scores[i]:
                        populations[k][i] = trial
                        personal_best_positions[k][i] = trial
                        personal_best_scores[k][i] = trial_score

            # Multi-swarm cooperation
            if evaluations < self.budget:
                center_positions = np.mean(np.array(personal_best_positions), axis=1)
                for k in range(self.num_swarms):
                    r3 = np.random.rand(self.population_size, self.dim)
                    populations[k] += r3 * (center_positions.mean(axis=0) - populations[k])

        return global_best_position, global_best_score