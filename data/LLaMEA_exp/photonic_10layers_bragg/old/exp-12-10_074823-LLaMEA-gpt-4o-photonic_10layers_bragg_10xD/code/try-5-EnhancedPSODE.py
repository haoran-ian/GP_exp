import numpy as np

class EnhancedPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w_max = 0.9  # maximum inertia weight
        self.w_min = 0.4  # minimum inertia weight
        self.F = 0.5  # differential weight
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
            self.de(mut_func=func, lb=lb, ub=ub)
        
        return self.global_best_position

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(low=lb, high=ub, size=(self.population_size, self.dim))
        self.velocity = np.random.uniform(low=-abs(ub - lb), high=abs(ub - lb), size=(self.population_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.array([float('inf')] * self.population_size)
        self.global_best_position = np.copy(self.population[np.argmin(self.personal_best_value)])
        self.global_best_value = min(self.personal_best_value)

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
            
            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
            
            # Update velocity and position
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocity[i] = (w * self.velocity[i] +
                                self.c1 * r1 * (self.personal_best_position[i] - self.population[i]) +
                                self.c2 * r2 * (self.global_best_position - self.population[i]))
            self.population[i] += self.velocity[i]
            
            # Clamp the position within bounds
            self.population[i] = np.clip(self.population[i], lb, ub)

    def de(self, mut_func, lb, ub):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break

            # Select three random individuals different from i using chaotic maps
            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = self.chaotic_selection(candidates)
            
            # Create a mutant vector
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
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

    def chaotic_selection(self, candidates):
        np.random.shuffle(candidates)
        return candidates[:3]