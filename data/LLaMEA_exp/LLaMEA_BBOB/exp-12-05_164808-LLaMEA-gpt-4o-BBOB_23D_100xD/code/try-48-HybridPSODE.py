import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.c1 = 1.5  # Cognitive parameter for PSO
        self.c2 = 1.5  # Social parameter for PSO
        self.w = 0.5   # Inertia weight for PSO
        self.F = 0.8   # Differential mutation factor for DE
        self.CR = 0.9  # Crossover rate for DE

    def __call__(self, func):
        np.random.seed(42)
        # Initialize particles and their velocities
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = func(global_best)
        
        eval_count = self.population_size
        while eval_count < self.budget:
            # PSO Update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.w = 0.4 * (1 - eval_count / self.budget) + 0.3 * np.random.rand()  # Adaptive inertia weight adjustment
                self.c1 = (1.7 * np.exp(-eval_count / self.budget)) + 0.3  # Exponential decay applied to cognitive parameter
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - particles[i]) +
                                 self.c2 * r2 * (global_best - particles[i]))
                particles[i] += velocities[i] + 0.1 * np.random.randn(self.dim)  # Enhanced exploration mechanism
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)
                
                fitness = func(particles[i])
                eval_count += 1
                if fitness < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = particles[i], fitness
                    if fitness < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = fitness
            
            if eval_count >= self.budget:
                break

            # DE Update (only if budget allows further evaluations)
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                a, b, c = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                adaptive_F = 0.6 + 0.4 * np.random.rand()  # Adaptive mutation factor
                mutant_vector = personal_best[a] + adaptive_F * (personal_best[b] - personal_best[c])
                adaptive_CR = 0.95 - 0.4 * (eval_count / self.budget)  # Adaptive crossover rate adjustment
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < adaptive_CR else particles[i, j]
                                         for j in range(self.dim)])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = trial_vector, trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial_vector
                        global_best_fitness = trial_fitness

        return global_best