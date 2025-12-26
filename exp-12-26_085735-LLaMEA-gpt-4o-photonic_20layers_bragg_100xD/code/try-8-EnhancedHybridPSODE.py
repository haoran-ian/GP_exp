import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.cross_prob = 0.7
        self.curr_evaluations = 0
        
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
            inertia_weight = self.inertia_weight_max - (
                (self.inertia_weight_max - self.inertia_weight_min) *
                (self.curr_evaluations / self.budget)
            )
            
            for i in range(self.pop_size):
                # PSO Update with adaptive inertia weight
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = (inertia_weight * vel[i] + 
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

                # Opposition-based learning
                opp_vector = lb + ub - trial_vector
                opp_vector = np.clip(opp_vector, lb, ub)
                
                # Evaluate and Select
                trial_score = func(trial_vector)
                opp_score = func(opp_vector)
                self.curr_evaluations += 2
                
                if trial_score < swarm_scores[i]:
                    swarm[i] = trial_vector
                    swarm_scores[i] = trial_score
                if opp_score < swarm_scores[i]:
                    swarm[i] = opp_vector
                    swarm_scores[i] = opp_score
                
                if swarm_scores[i] < pbest_scores[i]:
                    pbest_scores[i] = swarm_scores[i]
                    pbest_positions[i] = swarm[i]

                    if swarm_scores[i] < pbest_scores[gbest_idx]:
                        gbest_position = swarm[i]
                        gbest_idx = i

                if self.curr_evaluations >= self.budget:
                    break

        return gbest_position, pbest_scores[gbest_idx]