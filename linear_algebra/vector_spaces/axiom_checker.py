import numpy as np
from linear_algebra.vectors.vector_engine import VectorEngine

class AxiomExplorer:
    @staticmethod
    def explore_axioms(u: VectorEngine, v: VectorEngine, w: VectorEngine, c: float, d: float):
        """Explores standard vector space axioms using provided test vectors."""
        if u.dim != v.dim or v.dim != w.dim:
            raise ValueError("Axiom exploration requires vectors of the same dimension.")
            
        u_arr = np.array(u.vector)
        v_arr = np.array(v.vector)
        w_arr = np.array(w.vector)
        
        # 1. Commutativity: u + v == v + u
        comm = np.allclose(u_arr + v_arr, v_arr + u_arr)
        
        # 2. Associativity: (u + v) + w == u + (v + w)
        assoc = np.allclose((u_arr + v_arr) + w_arr, u_arr + (v_arr + w_arr))
        
        # 3. Distributivity over vector addition: c(u + v) == cu + cv
        dist_vec = np.allclose(c * (u_arr + v_arr), (c * u_arr) + (c * v_arr))
        
        # 4. Distributivity over scalar addition: (c + d)u == cu + du
        dist_sca = np.allclose((c + d) * u_arr, (c * u_arr) + (d * u_arr))
        
        return {
            "Commutativity ($u + v = v + u$)": comm,
            "Associativity ($(u + v) + w = u + (v + w)$)": assoc,
            "Distributivity of Scalar $c(u+v) = cu + cv$": dist_vec,
            "Distributivity over Scalars $(c+d)u = cu + du$": dist_sca
        }
