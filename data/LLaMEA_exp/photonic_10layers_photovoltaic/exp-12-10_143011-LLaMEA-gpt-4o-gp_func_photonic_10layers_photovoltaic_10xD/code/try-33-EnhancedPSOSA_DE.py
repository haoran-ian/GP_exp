import numpy as np

class EnhancedPSOSA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.F = 0.8  # DE mutation factor
        self.CR = 0.9  # DE crossover probability
        self.temp_factor = 0.99  # Cooling factor for Simulated Annealing
        self.particles = np.random.rand(self.pop_size, dim)
        self.velocities = np.random.rand(self.pop_size, dim) * 0.1
        self.best_personal_positions = np.copy(self.particles)
        self.best_personal_scores = np.full(self.pop_size, np.inf)
        self.best_global_position = np.zeros(dim)
        self.best_global_score = np.inf
        self.func_eval_count = 0

    def __call__(self, func):
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        # Normalize particles to the function's bounds
        self.particles = bounds_lb + self.particles * (bounds_ub - bounds_lb)
        
        while self.func_eval_count < self.budget:
            for i in range(self.pop_size):
                current_score = func(self.particles[i])
                self.func_eval_count += 1

                # Update personal best
                if current_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = current_score
                    self.best_personal_positions[i] = self.particles[i].copy()

                # Update global best
                if current_score < self.best_global_score:
                    self.best_global_score = current_score
                    self.best_global_position = self.particles[i].copy()

            # Update velocities and positions for PSO
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.best_personal_positions - self.particles) +
                               self.c2 * r2 * (self.best_global_position - self.particles))
            self.particles += self.velocities
            
            # Simulated Annealing inspired perturbation
            temp = self.temp_factor ** (self.func_eval_count / self.budget)
            perturbation = np.random.normal(0, temp, (self.pop_size, self.dim))
            self.particles += perturbation

            # Differential Evolution mutation and crossover
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds_lb, bounds_ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.particles[i])
                trial_score = func(trial)
                self.func_eval_count += 1

                if trial_score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = trial_score
                    self.best_personal_positions[i] = trial.copy()

                if trial_score < self.best_global_score:
                    self.best_global_score = trial_score
                    self.best_global_position = trial.copy()

            # Keep particles inside bounds
            self.particles = np.clip(self.particles, bounds_lb, bounds_ub)

        return self.best_global_position, self.best_global_score