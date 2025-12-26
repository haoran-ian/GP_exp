import numpy as np

class ImprovedHybridPSODE:
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

    def adapt_parameters(self):
        # Adaptive inertia weight decreasing linearly
        self.inertia_weight = 0.85 - 0.4 * (self.curr_evaluations / self.budget)  # slightly refined
        # Adaptive mutation factor based on success rate
        if len(self.archive) > 0:
            success_rate = sum(1 for _, succ in self.archive if succ) / len(self.archive)
            self.mutation_factor = 0.6 + 0.3 * success_rate  # slightly refined

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb + (ub - lb) * 0.1, ub - (ub - lb) * 0.1, (self.pop_size, self.dim))  # refined initialization
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
            self.adapt_parameters()
            for i in range(self.pop_size):
                # PSO Update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = (self.inertia_weight * vel[i] + 
                          self.cognitive_coeff * r1 * (pbest_positions[i] - swarm[i]) +
                          self.social_coeff * r2 * (gbest_position - swarm[i]))
                swarm[i] += vel[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # DE Mutation and Crossover
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.cross_prob
                trial_vector = np.where(crossover, mutant_vector, swarm[i])

                # Evaluate and Select
                trial_score = func(trial_vector)
                self.curr_evaluations += 1

                if trial_score < swarm_scores[i]:
                    self.archive.append((i, True))  # Archive success
                    swarm[i] = trial_vector
                    swarm_scores[i] = trial_score
                    
                    if trial_score < pbest_scores[i]:
                        pbest_scores[i] = trial_score
                        pbest_positions[i] = trial_vector

                        if trial_score < pbest_scores[gbest_idx]:
                            gbest_position = trial_vector
                            gbest_idx = i

                else:
                    self.archive.append((i, False))  # Archive failure

                if self.curr_evaluations >= self.budget:
                    break

        return gbest_position, pbest_scores[gbest_idx]