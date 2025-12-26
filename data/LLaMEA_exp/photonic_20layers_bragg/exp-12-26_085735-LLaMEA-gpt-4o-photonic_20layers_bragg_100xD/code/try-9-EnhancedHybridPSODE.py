import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.mutation_factor = 0.8
        self.cross_prob = 0.9
        self.curr_evaluations = 0
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim)) * (ub - lb)
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
            adaptive_inertia = 0.9 - 0.5 * (self.curr_evaluations / self.budget)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = (adaptive_inertia * vel[i] + 
                          self.cognitive_coeff * r1 * (pbest_positions[i] - swarm[i]) +
                          self.social_coeff * r2 * (gbest_position - swarm[i]))
                swarm[i] += vel[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cross_prob
                trial_vector = np.where(crossover, mutant_vector, swarm[i])

                trial_score = func(trial_vector)
                self.curr_evaluations += 1

                if trial_score < swarm_scores[i]:
                    swarm[i] = trial_vector
                    swarm_scores[i] = trial_score
                    
                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector

                        if trial_score < pbest_scores[gbest_idx]:
                            gbest_position = trial_vector
                            gbest_idx = i

                if self.curr_evaluations >= self.budget:
                    break

            # Local Search Phase
            if self.curr_evaluations < self.budget and np.random.rand() < 0.1:
                for _ in range(5):
                    local_search = gbest_position + np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                    local_search = np.clip(local_search, lb, ub)
                    local_score = func(local_search)
                    self.curr_evaluations += 1

                    if local_score < pbest_scores[gbest_idx]:
                        gbest_position = local_search
                        pbest_scores[gbest_idx] = local_score

                    if self.curr_evaluations >= self.budget:
                        break

        return gbest_position, pbest_scores[gbest_idx]