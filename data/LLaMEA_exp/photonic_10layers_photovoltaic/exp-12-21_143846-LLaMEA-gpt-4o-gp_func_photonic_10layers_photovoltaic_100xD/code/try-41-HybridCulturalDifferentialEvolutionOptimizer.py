import numpy as np

class HybridCulturalDifferentialEvolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.final_population_size = 20
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.5
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.elite_fraction = 0.1
        self.initial_mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.belief_space = {
            'normative': np.zeros((2, dim)),
            'situational': np.full(dim, np.inf)
        }
        
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
            inertia_weight = self.inertia_weight_final + (self.inertia_weight_initial - self.inertia_weight_final) * (1 - eval_count / self.budget)
            mutation_factor = self.initial_mutation_factor * (1 - eval_count / self.budget)  # Dynamic mutation factor
            
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

            elite_size = int(self.elite_fraction * current_population_size)
            elite_indices = np.argsort(personal_best_scores)[:elite_size]
            elite_positions = personal_best_positions[elite_indices]

            for d in range(self.dim):
                self.belief_space['normative'][0, d] = np.min(personal_best_positions[:, d])
                self.belief_space['normative'][1, d] = np.max(personal_best_positions[:, d])

            for i in range(current_population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_coeff * r1 * (personal_best_positions[i] - pop[i])
                social_component = self.social_coeff * r2 * (global_best_position - pop[i])
                velocities[i] = (inertia_weight * velocities[i] + cognitive_component + social_component)
                noise = np.random.normal(0, 0.1, self.dim)
                
                if i not in elite_indices:
                    elite_influence = np.mean(elite_positions - pop[i], axis=0)
                    velocities[i] += 0.1 * elite_influence
                    velocities[i] = np.clip(velocities[i], -1, 1)
                    
                # Differential Evolution Mutation and Crossover
                indices = np.random.choice(current_population_size, 3, replace=False)
                a, b, c = pop[indices[0]], pop[indices[1]], pop[indices[2]]
                mutant_vector = a + mutation_factor * (b - c)
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, pop[i])
                
                new_position = pop[i] + velocities[i] + noise
                pop[i] = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        
        return global_best_position, global_best_score