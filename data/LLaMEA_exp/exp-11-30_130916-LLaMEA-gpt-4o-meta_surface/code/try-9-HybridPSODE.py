import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(30, budget)
        self.initial_population_size = self.population_size
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9
        self.F = 0.8
        self.CR = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
    
    def __call__(self, func):
        eval_count = 0

        def evaluate_population():
            nonlocal eval_count
            scores = np.apply_along_axis(func, 1, self.population)
            eval_count += self.population_size
            for i in range(self.population_size):
                if scores[i] < self.pbest_scores[i]:
                    self.pbest_scores[i] = scores[i]
                    self.pbest_positions[i] = self.population[i]
                if scores[i] < self.gbest_score:
                    self.gbest_score = scores[i]
                    self.gbest_position = self.population[i]

        evaluate_population()

        while eval_count < self.budget:
            # PSO Update with adaptive parameters
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.w = 0.9 - (eval_count / self.budget) * 0.4  # Inertia weight
            self.c1 = 1.5 + (eval_count / self.budget) * 0.5  # Increase cognitive component
            self.c2 = 1.5 - (eval_count / self.budget) * 0.5  # Decrease social component
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.pbest_positions - self.population) +
                self.c2 * r2 * (self.gbest_position - self.population)
            )
            self.population += self.velocities
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])

            # Dynamic population size adjustment
            new_population_size = int(self.initial_population_size * (1 - eval_count / self.budget))
            if new_population_size < 5:
                new_population_size = 5
            if new_population_size != self.population_size:
                self.population = self.population[:new_population_size]
                self.velocities = self.velocities[:new_population_size]
                self.pbest_positions = self.pbest_positions[:new_population_size]
                self.pbest_scores = self.pbest_scores[:new_population_size]
                self.population_size = new_population_size

            # DE Update with adaptive mutation
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = self.F + 0.2 * np.random.rand() * (1 - eval_count / self.budget)
                mutant = np.clip(a + F_adaptive * (b - c), self.bounds[0], self.bounds[1])
                self.CR = 0.9 - (eval_count / self.budget) * 0.4  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                score_trial = func(trial)
                eval_count += 1
                if score_trial < self.pbest_scores[i]:
                    self.population[i] = trial
                    self.pbest_scores[i] = score_trial
                    self.pbest_positions[i] = trial
                    if score_trial < self.gbest_score:
                        self.gbest_score = score_trial
                        self.gbest_position = trial
                if eval_count >= self.budget:
                    break
        
        return self.gbest_position, self.gbest_score