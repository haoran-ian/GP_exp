import numpy as np

class EnhancedAdaptiveCulturalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.final_population_size = 20
        self.num_swarms = 3  # New parameter: number of swarms
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.5
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.belief_space = {
            'normative': np.zeros((2, dim)),
            'situational': np.full(dim, np.inf)
        }
        
    def __call__(self, func):
        eval_count = 0
        global_best_position = np.full(self.dim, np.inf)
        global_best_score = np.inf
        personal_best_scores = [np.full(self.initial_population_size, np.inf) for _ in range(self.num_swarms)]
        pops = [np.random.uniform(func.bounds.lb, func.bounds.ub, (self.initial_population_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.random.uniform(-1, 1, (self.initial_population_size, self.dim)) for _ in range(self.num_swarms)]
        personal_best_positions = [np.copy(pops[i]) for i in range(self.num_swarms)]

        while eval_count < self.budget:
            for swarm in range(self.num_swarms):
                current_population_size = int(
                    self.final_population_size + ((self.budget - eval_count) / self.budget) * 
                    (self.initial_population_size - self.final_population_size)
                )
                inertia_weight = ((self.budget - eval_count) / self.budget) * (self.inertia_weight_initial - self.inertia_weight_final) + self.inertia_weight_final
                
                for i in range(current_population_size):
                    score = func(pops[swarm][i])
                    eval_count += 1
                    if score < personal_best_scores[swarm][i]:
                        personal_best_scores[swarm][i] = score
                        personal_best_positions[swarm][i] = pops[swarm][i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = pops[swarm][i].copy()
                        self.belief_space['situational'] = global_best_position
                    
                    if eval_count >= self.budget:
                        return global_best_position, global_best_score

                for d in range(self.dim):
                    self.belief_space['normative'][0, d] = np.min([np.min(personal_best_positions[swarm][:, d]) for swarm in range(self.num_swarms)])
                    self.belief_space['normative'][1, d] = np.max([np.max(personal_best_positions[swarm][:, d]) for swarm in range(self.num_swarms)])

                for i in range(current_population_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[swarm][i] - pops[swarm][i])
                    social_component = self.social_coeff * r2 * (global_best_position - pops[swarm][i])
                    velocities[swarm][i] = (inertia_weight * velocities[swarm][i] + cognitive_component + social_component)
                    new_position = pops[swarm][i] + velocities[swarm][i]
                    pops[swarm][i] = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        
        return global_best_position, global_best_score