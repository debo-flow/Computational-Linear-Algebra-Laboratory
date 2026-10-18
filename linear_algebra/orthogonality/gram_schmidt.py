import numpy as np
from linear_algebra.vectors.vector_engine import VectorEngine
from .inner_product import InnerProductEngine

class GramSchmidtEngine:
    @staticmethod
    def compute(vectors, method='modified', tol=1e-9):
        """
        Computes an orthonormal basis from a set of vectors.
        Supports Classical Gram-Schmidt ('classical') and Modified Gram-Schmidt ('modified').
        Handles linearly dependent vectors by safely skipping them.
        """
        if not vectors:
            return {"orthonormal_vectors": [], "skipped_indices": [], "reports": []}
            
        is_complex = any(np.iscomplexobj(v.vector) for v in vectors)
        dtype = complex if is_complex else float
        
        Q = []
        skipped = []
        reports = []
        
        for j, v in enumerate(vectors):
            u_j = np.array(v.vector, dtype=dtype)
            coeffs = []
            
            if method == 'classical':
                # Classical: Project original vector v_j onto all previously computed q_i
                v_arr = np.array(v.vector, dtype=dtype)
                for q in Q:
                    q_arr = np.array(q.vector, dtype=dtype)
                    proj_coeff = np.vdot(q_arr, v_arr)  # <q, v>
                    coeffs.append(proj_coeff)
                    u_j -= proj_coeff * q_arr
            else:
                # Modified: Iteratively project the updated u_j onto q_i to reduce numerical drift
                for q in Q:
                    q_arr = np.array(q.vector, dtype=dtype)
                    proj_coeff = np.vdot(q_arr, u_j)  # <q, u_j>
                    coeffs.append(proj_coeff)
                    u_j -= proj_coeff * q_arr
                    
            u_norm = np.linalg.norm(u_j)
            
            # Linear Dependence Check
            if u_norm < tol:
                skipped.append(j)
                reports.append({
                    "index": j, "status": "skipped",
                    "reason": f"Vector {j} linearly dependent (residual norm {u_norm:.2e} < {tol})."
                })
                continue
                
            q_new = u_j / u_norm
            Q.append(VectorEngine(q_new))
            reports.append({
                "index": j, "status": "retained",
                "norm_before_normalization": u_norm,
                "projection_coefficients": coeffs
            })
            
        # Post-computation stability check
        ortho_check = InnerProductEngine.analyze_orthonormal_set(Q, tol)
        
        return {
            "orthonormal_vectors": Q,
            "skipped_indices": skipped,
            "reports": reports,
            "orthogonality_error": ortho_check['max_orthogonality_error']
        }
