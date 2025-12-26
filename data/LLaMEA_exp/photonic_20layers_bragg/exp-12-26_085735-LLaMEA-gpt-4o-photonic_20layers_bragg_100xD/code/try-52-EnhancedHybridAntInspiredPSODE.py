import numpy as np

class EnhancedHybridAntInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.cross_prob = 0.7
        self.curr_evaluations = 0
        self.archive = []
        self.vel_clamp = 0.1
        self.leader_follower_ratio = 0.2
        self.pheromone = np.ones(self.pop_size)
        
        # New: Tracking success rate and adaptive neighborhood
        self.neighborhood_size = max(3, self.pop_size // 10)
        self.success_rate = np.zeros(self.pop_size)

    def adapt_parameters(self, diversity):
        self.inertia_weight = 0.85 - 0.4 * (self.curr_evaluations / self.budget)
        if len(self.archive) > 0:
            success_rate = sum(1 for _, succ in self.archive if succ) / len(self.archive)
            self.mutation_factor = 0.6 + 0.3 * success_rate
            self.cross_prob = 0.6 + 0.2 * success_rate
        
        avg_pheromone = np.mean(self.pheromone)
        self.cognitive_coeff = 1.5 + 0.5 * (diversity / (ub - lb).mean())
        self.social_coeff = 1.3 + 0.2 * (1.0 - diversity / (ub - lb).mean()) + 0.1 * avg_pheromone

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb + (ub - lb) * 0.1, ub - (ub - lb) * 0.1, (self.pop_size, self.dim))
        vel = np.zeros((self.pop_size, self.dim))
        pbest_positions = np.copy(swarm)
        pbest_scores = np.full(self.pop_size, np.inf)

        swarm_scores = np.apply_along_axis(func, 1, swarm)
        self.curr_evaluations += self.pop_size
        
        for i in range(self.pop_size):
            if swarm_scores[i] < pbest_scores[i]:
                pbest_scores[i] = swarm_scores[i]
                pbest_positions[i] = swarm[i]

        gbest_idx = np.argmin(pbest_scores)
        gbest_position = pbest_positions[gbest_idx]

        while self.curr_evaluations < self.budget:
            diversity = np.std(swarm, axis=0).mean()
            self.adapt_parameters(diversity)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

                leader_follower = np.random.rand() < (self.leader_follower_ratio + self.pheromone[i])
                if leader_follower:
                    neighborhood = np.random.choice(self.pop_size, self.neighborhood_size, replace=False)
                    target_position = swarm[np.argmin([swarm_scores[n] for n in neighborhood])]
                else:
                    target_position = gbest_position

                vel[i] = (self.inertia_weight * vel[i] +
                          (self.cognitive_coeff * r1 + self.social_coeff * r2) * (target_position - swarm[i]) +
                          0.1 * (np.random.rand(self.dim) - 0.5))
                vel[i] = np.clip(vel[i], -self.vel_clamp, self.vel_clamp)
                swarm[i] += vel[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cross_prob
                greedy_vector = np.where(swarm_scores[i] < pbest_scores[i], swarm[i], mutant_vector)
                trial_vector = np.where(crossover, greedy_vector, swarm[i])

                trial_score = func(trial_vector)
                self.curr_evaluations += 1

                if trial_score < swarm_scores[i]:
                    self.archive.append((i, True))
                    swarm[i] = trial_vector
                    swarm_scores[i] = trial_score
                    self.pheromone[i] *= 1.1

                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector

                        if trial_score < pbest_scores[gbest_idx]:
                            gbest_position = trial_vector
                            gbest_idx = i

                else:
                    self.archive.append((i, False))
                    self.pheromone[i] *= 0.9

                if self.curr_evaluations >= self.budget:
                    break

            if self.curr_evaluations % (self.budget // 10) == 0:
                self.vel_clamp = np.std(swarm, axis=0).mean() * 0.15

        return gbest_position, pbest_scores[gbest_idx]