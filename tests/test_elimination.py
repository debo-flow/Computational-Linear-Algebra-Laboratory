import pytest
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.elimination import EliminationEngine

def test_gaussian_elimination_steps():
    A = MatrixEngine([[2, 1], [4, 3]])
    B = MatrixEngine([[1], [3]])
    sys = LinearSystem(A, B)
    
    steps = EliminationEngine.gaussian_elimination_steps(sys)
    # Step 0: Initial
    # Step 1: Scale R1 by 1/2
    # Step 2: Eliminate R2
    # The final step should have a 0 in the bottom left
    final_matrix_data = steps[-1]['matrix'].matrix.tolist()
    assert float(final_matrix_data[1][0]) == 0.0
