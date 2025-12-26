import numpy as np

class HPSO_ADE_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.w = 0.9  # Dynamic initial inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.c1 = 1.5  # Cognitive (personal) weight
        self.c2 = 1.5  # Social weight
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.adaptive_c1 = lambda t, T: 2.5 - 1.5 * (t / T)  # Adaptive cognitive weight
        self.adaptive_c2 = lambda t, T: 0.5 + 1.5 * (t / T)  # Adaptive social weight
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        self.budget -= self.population_size

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = np.copy(personal_best_positions[global_best_idx])
        global_best_score = personal_best_scores[global_best_idx]
        
        iteration = 0
        max_iterations = self.budget // self.population_size
        
        while self.budget > 0:
            self.w = self.w_min + (0.9 - self.w_min) * ((max_iterations - iteration) / max_iterations)
            c1 = self.adaptive_c1(iteration, max_iterations)
            c2 = self.adaptive_c2(iteration, max_iterations)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities + 
                          c1 * r1 * (personal_best_positions - population) +
                          c2 * r2 * (global_best_position - population))
            trial_population = population + velocities
            trial_population = np.clip(trial_population, lb, ub)
            
            for i in range(self.population_size):
                if np.random.rand() < self.cr:
                    indices = np.random.choice(self.population_size, 5, replace=False)
                    a, b, c, d, e = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]], population[indices[4]]
                    adaptive_f = self.f * (1 - (iteration / max_iterations)) * 0.5  # Refined adaptive differential weight
                    mutant = np.clip(a + adaptive_f * (b - c + d - e), lb, ub)
                    j_rand = np.random.randint(self.dim)
                    trial_population[i] = np.array([mutant[j] if np.random.rand() < self.cr or j == j_rand else trial_population[i][j] for j in range(self.dim)])
            
            trial_scores = np.array([func(ind) for ind in trial_population])
            self.budget -= self.population_size
            
            improvement = trial_scores < personal_best_scores
            personal_best_scores[improvement] = trial_scores[improvement]
            personal_best_positions[improvement] = trial_population[improvement]
            
            best_trial_idx = np.argmin(trial_scores)
            if trial_scores[best_trial_idx] < global_best_score:
                global_best_score = trial_scores[best_trial_idx]
                global_best_position = np.copy(trial_population[best_trial_idx])
            
            # Elite Strategy: Retain the best individual from both populations
            elite_idx = np.argmin(np.hstack((trial_scores, personal_best_scores)))
            if elite_idx < self.population_size:
                global_best_position = np.copy(trial_population[elite_idx])
            
            iteration += 1
            
        return global_best_position