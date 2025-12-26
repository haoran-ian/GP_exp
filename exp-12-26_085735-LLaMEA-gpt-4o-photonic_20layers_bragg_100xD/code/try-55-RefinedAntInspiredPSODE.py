import numpy as np

class RefinedAntInspiredPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight_init = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor_init = 0.8
        self.cross_prob_init = 0.7
        self.curr_evaluations = 0
        self.archive = []
        self.vel_clamp = 0.1
        self.leader_follower_ratio = 0.2
        self.pheromone = np.ones(self.pop_size)
        self.evaporate_rate = 0.9
        self.boost_factor = 1.2

    def adapt_parameters(self, diversity):
        progress = self.curr_evaluations / self.budget
        self.inertia_weight = (self.inertia_weight_init - self.inertia_weight_final) * (1 - progress) + self.inertia_weight_final
        self.mutation_factor = self.mutation_factor_init * (1 - progress)
        self.cross_prob = self.cross_prob_init * (1 - progress)

        success_rate = sum(1 for _, succ in self.archive if succ) / len(self.archive) if len(self.archive) > 0 else 0
        self.cognitive_coeff = 1.5 + 0.5 * success_rate
        self.social_coeff = 1.5 + 0.5 * (1 - success_rate) + 0.1 * np.mean(self.pheromone)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.pop_size, self.dim))
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
                target_position = gbest_position if not leader_follower else swarm[np.random.choice(range(self.pop_size))]

                vel[i] = (self.inertia_weight * vel[i] +
                          self.cognitive_coeff * r1 * (pbest_positions[i] - swarm[i]) +
                          self.social_coeff * r2 * (target_position - swarm[i]))
                vel[i] = np.clip(vel[i], -self.vel_clamp, self.vel_clamp)
                swarm[i] += vel[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                a, b, c = swarm[np.random.choice(range(self.pop_size), 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cross_prob
                trial_vector = np.where(crossover, mutant_vector, swarm[i])

                trial_score = func(trial_vector)
                self.curr_evaluations += 1

                if trial_score < swarm_scores[i]:
                    self.archive.append((i, True))
                    swarm[i] = trial_vector
                    swarm_scores[i] = trial_score
                    self.pheromone[i] *= self.boost_factor

                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector

                        if trial_score < pbest_scores[gbest_idx]:
                            gbest_position = trial_vector
                            gbest_idx = i

                else:
                    self.archive.append((i, False))
                    self.pheromone[i] *= self.evaporate_rate

                if self.curr_evaluations >= self.budget:
                    break

            self.vel_clamp = np.std(swarm, axis=0).mean() * 0.15

        return gbest_position, pbest_scores[gbest_idx]