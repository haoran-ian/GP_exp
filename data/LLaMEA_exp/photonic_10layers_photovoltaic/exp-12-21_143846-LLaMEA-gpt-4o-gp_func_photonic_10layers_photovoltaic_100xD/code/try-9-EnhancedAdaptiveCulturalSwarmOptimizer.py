import numpy as np

class EnhancedAdaptiveCulturalSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.final_population_size = 20
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.5
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.belief_space = {
            'normative': np.zeros((2, dim)),  # [0] for min, [1] for max
            'situational': np.full(dim, np.inf)  # Best solution found
        }
        
    def levy_flight(self, size, beta=1.5):
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / abs(v) ** (1 / beta)
        return step
    
    def __call__(self, func):
        eval_count = 0
        global_best_position = np.full(self.dim, np.inf)
        global_best_score = np.inf
        personal_best_scores = np.full(self.initial_population_size, np.inf)
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.initial_population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.initial_population_size, self.dim))
        personal_best_positions = np.copy(pop)

        while eval_count < self.budget:
            current_population_size = int(
                self.final_population_size + ((self.budget - eval_count) / self.budget) * 
                (self.initial_population_size - self.final_population_size)
            )
            inertia_weight = ((self.budget - eval_count) / self.budget) * (self.inertia_weight_initial - self.inertia_weight_final) + self.inertia_weight_final
            
            for i in range(current_population_size):
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

            for d in range(self.dim):
                self.belief_space['normative'][0, d] = np.min(personal_best_positions[:, d])
                self.belief_space['normative'][1, d] = np.max(personal_best_positions[:, d])

            for i in range(current_population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[i] - pop[i])
                social_component = self.social_coeff * r2 * (global_best_position - pop[i])
                velocities[i] = (inertia_weight * velocities[i] + cognitive_component + social_component)
                levy_mutation = self.levy_flight(self.dim) * (pop[i] - global_best_position)
                new_position = pop[i] + velocities[i] + levy_mutation
                pop[i] = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        
        return global_best_position, global_best_score