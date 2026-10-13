from linear_algebra.matrix.operations import MatrixOperations
from .system_model import LinearSystem

class SystemClassifier:
    @staticmethod
    def classify(system: LinearSystem, tol=1e-9):
        """Classifies the system using the Rouché-Capelli theorem."""
        rank_A = MatrixOperations.rank(system.A, tol)
        aug = system.augmented_matrix()
        rank_aug = MatrixOperations.rank(aug, tol)
        n_vars = system.A.cols

        if rank_A < rank_aug:
            status = "Inconsistent (No Solution)"
            solution_type = "none"
        elif rank_A == n_vars:
            status = "Consistent (Unique Solution)"
            solution_type = "unique"
        else:
            free_vars = n_vars - rank_A
            status = f"Consistent (Infinitely Many Solutions, {free_vars} free variable(s))"
            solution_type = "infinite"

        return {
            "rank_A": rank_A,
            "rank_aug": rank_aug,
            "unknowns": n_vars,
            "status": status,
            "solution_type": solution_type
        }
