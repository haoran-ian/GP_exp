import numpy as np

class AdaptiveInertiaQHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)  # Harmony memory size
        self.hmcr_init = 0.95  # Initial Harmony memory consideration rate
        self.hmcr_final = 0.70  # Final Harmony memory consideration rate
        self.par_init = 0.5  # Initial Pitch adjustment rate
        self.par_final = 0.05  # Final Pitch adjustment rate
        self.evaluations = 0
        self.quantum_prob = 0.20  # Initial Quantum probability
        self.learning_rate = 0.15  # Adaptive learning rate
        self.inertia_weight_init = 0.9  # Initial inertia weight
        self.inertia_weight_final = 0.4  # Final inertia weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.hms, self.dim))
        harmony_scores = np.array([func(harmony_memory[i]) for i in range(self.hms)])
        self.evaluations += self.hms

        best_harmony = harmony_memory[np.argmin(harmony_scores)]
        best_score = np.min(harmony_scores)

        while self.evaluations < self.budget:
            progress = self.evaluations / self.budget
            hmcr = self.hmcr_init - (self.hmcr_init - self.hmcr_final) * progress
            par = self.par_init - (self.par_init - self.par_final) * progress
            adaptive_quantum_prob = self.quantum_prob * (1 - progress) + 0.20 * progress

            inertia_weight = self.inertia_weight_init - (self.inertia_weight_init - self.inertia_weight_final) * progress

            new_harmony = np.array([harmony_memory[np.random.randint(self.hms), j] 
                                    if np.random.rand() < hmcr else np.random.uniform(lb[j], ub[j]) 
                                    for j in range(self.dim)])

            if np.random.rand() < par:
                dynamic_step = (1 - progress) * 0.1
                gradient_approximation = (ub - lb) / self.dim * (-1 + 2 * np.random.rand(self.dim))
                new_harmony += self.learning_rate * gradient_approximation * dynamic_step
                new_harmony = np.clip(new_harmony, lb, ub)

            if np.random.rand() < adaptive_quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(self.hms)])
                dynamic_delta_factor = np.sqrt(2) * (1 + progress)
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / dynamic_delta_factor
                new_harmony = inertia_weight * new_harmony + (1 - inertia_weight) * best_harmony
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony
                self.learning_rate = max(0.05, self.learning_rate * 0.98)

            worst_idx = np.argmax(harmony_scores)
            if score < harmony_scores[worst_idx]:
                harmony_memory[worst_idx], harmony_scores[worst_idx] = new_harmony, score

        return best_harmony, best_score