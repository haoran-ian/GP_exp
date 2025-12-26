import numpy as np
from scipy.optimize import minimize

class EnhancedDynamicChaoticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30  # Increased initial population size for better exploration
        self.final_population_size = 10  # Reduced final population size for better exploitation
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4  # Slightly higher minimum inertia for stability
        self.cognitive_component = 2.0
        self.social_component = 2.5  # Increased social component for better global search
        self.cr = 0.7
        self.chaotic_sequence = self._init_chaotic_sequence()

    def _init_chaotic_sequence(self):
        chaotic_sequence = np.zeros(self.budget)
        chaotic_sequence[0] = np.random.rand()
        r = 3.99  # Slightly increased chaotic parameter for more dynamic search
        for i in range(1, self.budget):
            chaotic_sequence[i] = r * chaotic_sequence[i-1] * (1 - chaotic_sequence[i-1])
        return chaotic_sequence

    def __call__(self, func):
        np.random.seed(42)
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub

        positions = np.random.uniform(lower_bound, upper_bound, (self.initial_population_size, self.dim))
        velocities = np.zeros((self.initial_population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.initial_population_size
        iteration = 0

        while evaluations < self.budget:
            inertial_range = self.inertia_weight_max - self.inertia_weight_min
            self.inertia_weight = self.inertia_weight_min + \
                                  inertial_range * (1 - evaluations / self.budget)
            chaotic_factor = self.chaotic_sequence[evaluations] * 0.5

            population_size = int(self.initial_population_size - 
                                  (self.initial_population_size - self.final_population_size) * 
                                  (evaluations / self.budget))

            for i in range(population_size):
                r1, r2 = np.random.rand(2, self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] + 
                                 chaotic_factor * (global_best_position - positions[i]) +
                                 self.cognitive_component * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_component * r2 * (global_best_position - positions[i]))
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)

            scores = np.array([func(x) for x in positions[:population_size]])
            evaluations += population_size

            for i in range(population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = positions[i]
                    if scores[i] < global_best_score:
                        global_best_score = scores[i]
                        global_best_position = positions[i]

            if evaluations < self.budget:
                best_idx = np.argmin(scores)
                res = minimize(func, positions[best_idx], bounds=[(lb, ub) for lb, ub in zip(lower_bound, upper_bound)], method='L-BFGS-B')
                if res.fun < global_best_score:
                    global_best_score = res.fun
                    global_best_position = res.x
                evaluations += res.nfev

        return global_best_position, global_best_score