import numpy as np

class EnhancedSubpopulationPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 + dim
        self.population_size = self.initial_population_size
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.velocity_clamp = 0.5 * (self.upper_bound - self.lower_bound)
        self.phase_change_point = self.budget // 3
        self.subpopulation_size = self.population_size // 2

    def __call__(self, func):
        evals = 0
        phase = 1
        subpopulations = [self.population]

        while evals < self.budget:
            for subpop in subpopulations:
                for i in range(len(subpop)):
                    score = func(subpop[i])
                    evals += 1

                    if score < self.pbest_scores[i]:
                        self.pbest_scores[i] = score
                        self.pbest_positions[i] = subpop[i]

                    if score < self.gbest_score:
                        self.gbest_score = score
                        self.gbest_position = subpop[i]

                if evals >= self.budget:
                    break

                if evals >= self.phase_change_point * phase:
                    phase += 1
                    self.c1 -= 0.5
                    self.c2 += 0.5

                self.w = 0.4 + 0.5 * (1 - evals / self.budget)

                for i in range(len(subpop)):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = self.c1 * r1 * (self.pbest_positions[i] - subpop[i])
                    social = self.c2 * r2 * (self.gbest_position - subpop[i])
                    self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                    self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                    subpop[i] = np.clip(subpop[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            if evals % (self.budget // 5) == 0:
                if np.ptp(self.pbest_scores) < 1e-6:
                    new_subpop = np.random.uniform(self.lower_bound, self.upper_bound, (self.subpopulation_size, self.dim))
                    subpopulations.append(new_subpop)
                    self.velocities = np.random.uniform(-1, 1, (self.subpopulation_size, self.dim))
                    self.pbest_positions = np.vstack([self.pbest_positions, new_subpop])
                    self.pbest_scores = np.concatenate([self.pbest_scores, np.full(self.subpopulation_size, np.inf)])

            if np.random.rand() < 0.01:
                for subpop in subpopulations:
                    subpop[:] = np.random.uniform(self.lower_bound, self.upper_bound, subpop.shape)
                    self.velocities = np.random.uniform(-1, 1, subpop.shape)
                    self.pbest_positions = np.copy(subpop)
                    self.pbest_scores.fill(np.inf)

        return self.gbest_position, self.gbest_score