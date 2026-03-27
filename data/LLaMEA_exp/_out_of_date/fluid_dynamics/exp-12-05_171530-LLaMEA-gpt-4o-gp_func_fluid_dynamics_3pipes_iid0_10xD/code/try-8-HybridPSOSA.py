import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30  
        self.velocity_clamp = (-1.0, 1.0)
        self.w_max = 0.9  
        self.w_min = 0.4  
        self.c1 = 1.7  
        self.c2_max = 1.5  
        self.cooling_rate = 0.99  
        self.temp = 1.0  

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size
        while evaluations < self.budget:
            self.w = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            self.c2 = self.c2_max * (1 - evaluations / self.budget) 
            
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (personal_best_positions[i] - particles[i])
                                 + self.c2 * r2 * (global_best_position - particles[i]))
                velocities[i] = np.clip(velocities[i], self.velocity_clamp[0], self.velocity_clamp[1])

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                score = func(particles[i])
                evaluations += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(particles[i])

                acceptance_probability = np.exp((global_best_score - score) / self.temp)
                if score < global_best_score or acceptance_probability > np.random.rand():
                    global_best_score = score
                    global_best_position = np.copy(particles[i])
                    
                # Dynamic parameter adaptation
                if evaluations % 10 == 0:  # Adjust parameters every 10 evaluations
                    successful_moves = np.sum(personal_best_scores < global_best_score)
                    self.c1 += 0.1 * (successful_moves / self.population_size)
                    self.c2 -= 0.1 * (successful_moves / self.population_size)

            self.temp *= self.cooling_rate

        return global_best_position, global_best_score