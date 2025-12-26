import numpy as np

class EnhancedAdaptiveCulturalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.5
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.belief_space = {
            'normative': np.zeros((2, dim)),  # [0] for min, [1] for max
            'situational': np.full(dim, np.inf)  # Best solution found
        }
        
    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = np.full(self.dim, np.inf)
        global_best_score = np.inf
        eval_count = 0
        
        while eval_count < self.budget:
            inertia_weight = ((self.budget - eval_count) / self.budget) * (self.inertia_weight_initial - self.inertia_weight_final) + self.inertia_weight_final
            
            for i in range(self.population_size):
                score = func(pop[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pop[i].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pop[i].copy()
                    self.belief_space['situational'] = global_best_position
                
                if eval_count >= self.budget:
                    break

            # Update belief space normative knowledge
            for d in range(self.dim):
                self.belief_space['normative'][0, d] = np.min(personal_best_positions[:, d])
                self.belief_space['normative'][1, d] = np.max(personal_best_positions[:, d])
                
            # Update particles
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[i] - pop[i])
                social_component = self.social_coeff * r2 * (global_best_position - pop[i])
                velocities[i] = (inertia_weight * velocities[i] + cognitive_component + social_component)
                
                # Add random noise for diversity preservation
                noise = np.random.normal(0, 0.1, self.dim)
                new_position = pop[i] + velocities[i] + noise
                pop[i] = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        
        return global_best_position, global_best_score