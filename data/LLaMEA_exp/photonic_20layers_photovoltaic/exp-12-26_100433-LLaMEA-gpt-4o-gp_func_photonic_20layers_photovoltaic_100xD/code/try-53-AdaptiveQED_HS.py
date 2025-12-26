import numpy as np

class AdaptiveQED_HS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.hms = min(50, budget // 10)  # Harmony memory size
        self.hmcr = 0.9  # Initial Harmony memory consideration rate
        self.par = 0.4  # Initial Pitch adjustment rate
        self.evaluations = 0
        self.alpha = 0.99  # Learning rate decay factor
        self.quantum_prob = 0.1

    def update_parameters(self, progress):
        self.hmcr = self.hmcr * (self.alpha ** progress)
        self.par = self.par * (self.alpha ** progress)
        self.quantum_prob = self.quantum_prob * (self.alpha ** progress) + 0.05 * (1 - self.alpha ** progress)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = np.random.uniform(lb, ub, (self.hms, self.dim))
        harmony_scores = np.full(self.hms, np.inf)

        best_harmony = None
        best_score = np.inf

        for i in range(self.hms):
            score = func(harmony_memory[i])
            self.evaluations += 1
            harmony_scores[i] = score
            if score < best_score:
                best_score = score
                best_harmony = harmony_memory[i]

        while self.evaluations < self.budget:
            progress = self.evaluations / self.budget
            self.update_parameters(progress)

            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(0, self.hms)
                    new_harmony[j] = harmony_memory[idx, j]
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-0.1, 0.1) * (ub[j] - lb[j])
                        new_harmony[j] = np.clip(new_harmony[j], lb[j], ub[j])
                else:
                    new_harmony[j] = np.random.uniform(lb[j], ub[j])

            if np.random.rand() < self.quantum_prob:
                center = (np.mean(harmony_memory, axis=0) + best_harmony) / 2
                delta = np.abs(best_harmony - harmony_memory[np.random.randint(0, self.hms)])
                dynamic_delta_factor = np.sqrt(2) * (1 + progress)
                new_harmony = center + np.random.uniform(-1, 1, self.dim) * delta / dynamic_delta_factor
                new_harmony = np.clip(new_harmony, lb, ub)

            score = func(new_harmony)
            self.evaluations += 1

            if score < best_score:
                best_score = score
                best_harmony = new_harmony

            worst_idx = np.argmax(harmony_scores)
            if score < harmony_scores[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                harmony_scores[worst_idx] = score

        return best_harmony, best_score