import numpy as np

class AdaptiveMultiPopulationPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.num_subpopulations = 2
        self.subpop_size = self.pop_size // self.num_subpopulations
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.cross_prob = 0.7
        self.curr_evaluations = 0
        self.archive = []
        self.vel_clamp = 0.1

    def adapt_parameters(self):
        self.inertia_weight = 0.85 - 0.4 * (self.curr_evaluations / self.budget)
        if len(self.archive) > 0:
            success_rate = sum(1 for _, succ in self.archive if succ) / len(self.archive)
            self.mutation_factor = 0.6 + 0.3 * success_rate
            self.cross_prob = 0.6 + 0.2 * success_rate

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        subpopulations = [np.random.uniform(lb, ub, (self.subpop_size, self.dim)) for _ in range(self.num_subpopulations)]
        velocities = [np.zeros((self.subpop_size, self.dim)) for _ in range(self.num_subpopulations)]
        pbest_positions = [np.copy(subpop) for subpop in subpopulations]
        pbest_scores = [np.full(self.subpop_size, np.inf) for _ in range(self.num_subpopulations)]
        
        subpop_scores = [np.apply_along_axis(func, 1, subpop) for subpop in subpopulations]
        self.curr_evaluations += self.pop_size

        for idx, scores in enumerate(subpop_scores):
            for i in range(self.subpop_size):
                if scores[i] < pbest_scores[idx][i]:
                    pbest_scores[idx][i] = scores[i]
                    pbest_positions[idx][i] = subpopulations[idx][i]

        gbest_position = None
        gbest_score = np.inf
        for idx in range(self.num_subpopulations):
            gbest_idx = np.argmin(pbest_scores[idx])
            if pbest_scores[idx][gbest_idx] < gbest_score:
                gbest_position = pbest_positions[idx][gbest_idx]
                gbest_score = pbest_scores[idx][gbest_idx]

        while self.curr_evaluations < self.budget:
            self.adapt_parameters()
            for subpop_idx, (subpop, vel, p_positions, p_scores) in enumerate(zip(subpopulations, velocities, pbest_positions, pbest_scores)):
                for i in range(self.subpop_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    vel[i] = (self.inertia_weight * vel[i] + 
                              self.cognitive_coeff * r1 * (p_positions[i] - subpop[i]) +
                              self.social_coeff * r2 * (gbest_position - subpop[i]))
                    vel[i] = np.clip(vel[i], -self.vel_clamp, self.vel_clamp)
                    subpop[i] += vel[i]
                    subpop[i] = np.clip(subpop[i], lb, ub)

                    idxs = [idx for idx in range(self.subpop_size) if idx != i]
                    a, b, c = subpop[np.random.choice(idxs, 3, replace=False)]
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                    crossover = np.random.rand(self.dim) < self.cross_prob
                    greedy_vector = np.where(subpop_scores[subpop_idx][i] < p_scores[i], subpop[i], mutant_vector)
                    trial_vector = np.where(crossover, greedy_vector, subpop[i])

                    trial_score = func(trial_vector)
                    self.curr_evaluations += 1

                    if trial_score < subpop_scores[subpop_idx][i]:
                        self.archive.append((i, True))
                        subpop[i] = trial_vector
                        subpop_scores[subpop_idx][i] = trial_score
                        
                        if trial_score < p_scores[i]:
                            p_scores[i] = trial_score
                            p_positions[i] = trial_vector

                            if trial_score < gbest_score:
                                gbest_position = trial_vector
                                gbest_score = trial_score

                    else:
                        self.archive.append((i, False))

                    if self.curr_evaluations >= self.budget:
                        break

                if self.curr_evaluations >= self.budget:
                    break

            if self.curr_evaluations % (self.budget // 10) == 0:
                self.vel_clamp = np.std([np.std(subpop, axis=0) for subpop in subpopulations], axis=0).mean() * 0.1

        return gbest_position, gbest_score