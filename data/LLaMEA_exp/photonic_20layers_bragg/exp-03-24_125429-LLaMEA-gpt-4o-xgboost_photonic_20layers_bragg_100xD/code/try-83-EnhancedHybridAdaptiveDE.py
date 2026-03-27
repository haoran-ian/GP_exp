import numpy as np
from scipy.stats import levy

class EnhancedHybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F_min = 0.4
        self.F_max = 0.95
        self.CR_min = 0.1
        self.CR_max = 0.9
        self.stochastic_tunneling_prob = 0.25
        self.adaptive_rate = 0.05
        self.noise_intensity = 0.05
        self.memory = []
        self.phase_detection_window = 10
        self.phase_transition_threshold = 0.1
        self.dynamic_pop_scale = 0.1

    def _initialize_population(self, bounds):
        pop = np.random.rand(self.population_size, self.dim)
        return bounds.lb + pop * (bounds.ub - bounds.lb)

    def _mutate(self, pop, idx, bounds):
        a, b, c = np.random.choice(np.delete(np.arange(self.population_size), idx), 3, replace=False)
        F = np.random.uniform(self.F_min, self.F_max)
        mutant = pop[a] + F * (pop[b] - pop[c])
        return np.clip(mutant, bounds.lb, bounds.ub)

    def _crossover(self, target, mutant):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        cross_points = np.random.rand(self.dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def _stochastic_tunneling(self, candidate, bounds):
        scale = np.random.uniform(0.05, 0.15)
        perturbed = candidate + scale * np.random.uniform(-1, 1, self.dim) * (bounds.ub - bounds.lb)
        return np.clip(perturbed, bounds.lb, bounds.ub)

    def _levy_flight(self, candidate, bounds):
        step = levy.rvs(size=self.dim) * (bounds.ub - bounds.lb) / 100
        levy_candidate = candidate + step
        return np.clip(levy_candidate, bounds.lb, bounds.ub)

    def _add_noise(self, candidate, bounds):
        noise = np.random.normal(0, self.noise_intensity, self.dim) * (bounds.ub - bounds.lb)
        noisy_candidate = candidate + noise
        return np.clip(noisy_candidate, bounds.lb, bounds.ub)

    def _diversity_control(self, pop, bounds):
        if len(self.memory) > self.phase_detection_window:
            diversity = np.std(self.memory[-self.phase_detection_window:], axis=0)
            if np.mean(diversity) < self.phase_transition_threshold:
                return self._initialize_population(bounds)
        return pop

    def _adaptive_landscape_detection(self):
        if len(self.memory) > self.phase_detection_window:
            fitness_changes = np.diff(self.memory[-self.phase_detection_window:])
            return np.std(fitness_changes) / np.mean(fitness_changes)
        return 0

    def _dynamic_population_size(self):
        return max(10, int(self.population_size + (self.dynamic_pop_scale * self.budget)))

    def __call__(self, func):
        bounds = func.bounds
        self.population_size = self._dynamic_population_size()
        pop = self._initialize_population(bounds)
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        for _ in range(self.budget - self.population_size):
            sensitivity = self._adaptive_landscape_detection()
            if sensitivity > self.phase_transition_threshold:
                self.stochastic_tunneling_prob = min(1.0, self.stochastic_tunneling_prob + self.adaptive_rate)

            for i in range(self.population_size):
                mutant = self._mutate(pop, i, bounds)
                trial = self._crossover(pop[i], mutant)

                if np.random.rand() < self.stochastic_tunneling_prob:
                    trial = self._stochastic_tunneling(trial, bounds)
                
                if np.random.rand() < 0.2:
                    trial = self._levy_flight(trial, bounds)

                if np.random.rand() < 0.5:
                    trial = self._add_noise(trial, bounds)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best):
                        best = trial

                    self.F_max = min(1.0, self.F_max + self.adaptive_rate * 0.1)
                    self.CR_max = min(1.0, self.CR_max + self.adaptive_rate * 0.1)
                else:
                    self.F_min = max(0.1, self.F_min - self.adaptive_rate)
                    self.CR_min = max(0.0, self.CR_min - self.adaptive_rate)

            self.memory.append(func(best))
            pop = self._diversity_control(pop, bounds)

        return best