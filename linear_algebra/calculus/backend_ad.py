import numpy as np
import warnings

class BackendAD:
    """
    Integrates external production-grade AutoDiff backends (PyTorch/JAX) if available.
    Degrades gracefully if not installed.
    """
    @staticmethod
    def check_pytorch():
        try:
            import torch
            return True, torch
        except ImportError:
            return False, None

    @staticmethod
    def torch_matrix_gradient(matrix_A_val, matrix_B_val):
        """
        Demonstrates advanced Matrix-Calculus AutoDiff finding gradient of f(A) = Tr(A^T @ B @ A).
        """
        is_avail, torch = BackendAD.check_pytorch()
        if not is_avail:
            return False, "PyTorch is not installed. CPU-backend AD is unavailable.", None
            
        # Define tensors with gradients required
        A = torch.tensor(matrix_A_val, dtype=torch.float64, requires_grad=True)
        B = torch.tensor(matrix_B_val, dtype=torch.float64, requires_grad=False)
        
        # f(A) = Trace(A^T B A)
        loss = torch.trace(A.t() @ B @ A)
        loss.backward()
        
        # Analytic gradient mathematically is (B + B^T)A
        grad_A = A.grad.detach().numpy()
        loss_val = loss.item()
        
        return True, loss_val, grad_A
