import numpy as np

class AdaptivePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7   # inertia weight
        self.F_max = 0.9  # maximum differential weight
        self.F_min = 0.3  # minimum differential weight
        self.CR = 0.9  # crossover probability
        self.population = None
        self.velocity = None
        self.personal_best_position = None
        self.personal_best_value = None
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        
        while self.evaluations < self.budget:
            self.update_particles(func, lb, ub)
            adaptive_F = self.adaptive_differential_weight()
            self.de(mut_func=func, lb=lb, ub=ub, F=adaptive_F)
        
        return self.global_best_position

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocity = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.array([float('inf')] * self.population_size)
    
    def update_particles(self, func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            # Evaluate the current solution
            fitness = func(self.population[i])
            self.evaluations += 1
            
            # Update personal best
            if fitness < self.personal_best_value[i]:
                self.personal_best_value[i] = fitness
                self.personal_best_position[i] = self.population[i].copy()
                
            # Update global best
            if fitness < self.global_best_value:
                self.global_best_value = fitness
                self.global_best_position = self.population[i].copy()
            
            # Update velocity and position
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            inertia_weight = np.random.uniform(0.5, 0.9) # Adaptive inertia weight to balance exploration and exploitation
            self.velocity[i] = (inertia_weight * self.velocity[i] +
                                self.c1 * r1 * (self.personal_best_position[i] - self.population[i]) +
                                self.c2 * r2 * (self.global_best_position - self.population[i]))
            self.population[i] += self.velocity[i]
            
            # Clamp the position within bounds
            self.population[i] = np.clip(self.population[i], lb, ub)

    def de(self, mut_func, lb, ub, F):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            # Select three random individuals different from i
            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            
            # Create a mutant vector
            mutant = self.population[a] + F * (self.population[b] - self.population[c])
            mutant = np.clip(mutant, lb, ub)

            # Perform crossover
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])

            # Evaluate the trial solution
            trial_fitness = mut_func(trial)
            self.evaluations += 1

            # Selection
            if trial_fitness < self.personal_best_value[i]:
                self.population[i] = trial
                self.personal_best_value[i] = trial_fitness
                self.personal_best_position[i] = trial.copy()
                
                # Update global best
                if trial_fitness < self.global_best_value:
                    self.global_best_value = trial_fitness
                    self.global_best_position = trial.copy()
    
    def adaptive_differential_weight(self):
        # Calculate an adaptive differential weight based on the current best solution progress
        progress_ratio = self.evaluations / self.budget
        return self.F_min + (self.F_max - self.F_min) * (1 - progress_ratio)