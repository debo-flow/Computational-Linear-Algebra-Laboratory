import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla

# Milestones 1-15 Engines
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.matrix.validation import ValidationEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix.advanced_operations import AdvancedMatrixOperations
from linear_algebra.matrix.decompositions import MatrixDecompositions
from linear_algebra.matrix.analysis import MatrixAnalysis
from linear_algebra.matrix.minors_cofactors import MinorsCofactors
from linear_algebra.matrix.singularity_analysis import SingularityAnalyzer
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.verification import SystemVerification
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.basis import BasisAnalyzer
from visualization.transformation_plots import plot_2d_transformation
from linear_algebra.subspaces.fundamental import FundamentalSubspaces
from linear_algebra.eigenvalues.core import EigenEngine
from linear_algebra.diagonalization.core import DiagonalizationEngine
from linear_algebra.svd.core import SVDEngine
from linear_algebra.svd.approximation import LowRankApproximation
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine
from visualization.svd_plots import plot_singular_values, plot_energy_retention
from linear_algebra.qr.core import QREngine
from linear_algebra.qr.least_squares import QRLeastSquares
from linear_algebra.factorizations.lu import LUEngine
from linear_algebra.factorizations.cholesky import CholeskyEngine
from linear_algebra.factorizations.triangular_solver import TriangularSolver
from linear_algebra.definiteness.classifier import DefinitenessClassifier
from linear_algebra.definiteness.sylvester import SylvesterCriterion
from linear_algebra.definiteness.quadratic_forms import QuadraticForms
from linear_algebra.definiteness.loewner import LoewnerOrder
from linear_algebra.definiteness.generator import DefiniteMatrixGenerator
from visualization.quadratic_plots import plot_2d_quadratic_form

# ================= NEW M16 IMPORTS =================
from linear_algebra.numerical_eigen.iterative import IterativeEigenEngine
from linear_algebra.numerical_eigen.qr_algorithm import QREigenAlgorithm
from linear_algebra.numerical_eigen.schur import SchurDecomposition
from linear_algebra.numerical_eigen.sensitivity import EigenSensitivity
from visualization.eigen_plots import plot_eigenvalue_spectrum, plot_convergence_history
# ==================================================

st.set_page_config(page_title="Comp. LinAlg Lab", layout="wide")

st.sidebar.title("Navigation")
pages = [
    "Home", 
    "Matrix Calculator", 
    "Advanced Operations",
    "Decompositions",
    "Determinants & Adjugate",      
    "Rank & Singularity",           
    "Linear Systems Lab",           
    "Vector Spaces Lab",            
    "Basis & Coordinates Lab",      
    "Linear Transformations Lab",   
    "Fundamental Subspaces Lab",    
    "Eigenvalues & Polynomials",    
    "Matrix Diagonalization Lab",
    "SVD & Low-Rank Approx Lab",
    "Orthogonality & Projections",
    "QR & Least-Squares Lab",
    "Advanced Factorizations Lab",
    "Pos-Def & Quadratic Forms",
    "Numerical Eigen & Schur Lab",  # NEW M16 TAB
    "Vector Calculator", 
    "Validation Lab"
]
choice = st.sidebar.radio("Go to", pages)

st.sidebar.markdown("---")
edu_mode = st.sidebar.checkbox("🎓 Enable Educational Mode", value=True)
precision = st.sidebar.slider("Numerical Tolerance (10^-x)", 1, 15, 9)
tol = 10**(-precision)

def parse_input(mat_str):
    try:
        clean_str = mat_str.strip().replace('[', '').replace(']', '')
        rows = clean_str.split(';')
        data = [[complex(val) if 'j' in val else float(val) for val in row.split(',')] for row in rows]
        return MatrixEngine(data)
    except Exception as e:
        st.error(f"Input Error: Please use format '1,2; 3,4'. Detail: {e}")
        return None

def parse_vector_list(text):
    try:
        vec_strs = text.split('|')
        return [VectorEngine([complex(x) if 'j' in x else float(x) for x in v.split(',')]) for v in vec_strs if v.strip()]
    except Exception:
        return None

def format_complex(val):
    if isinstance(val, (int, float, np.float64, np.float32)): return f"{val:.4g}"
    if isinstance(val, (complex, np.complex128)):
        if abs(val.imag) < 1e-9: return f"{val.real:.4g}"
        sign = "+" if val.imag >= 0 else "-"
        return f"{val.real:.4g} {sign} {abs(val.imag):.4g}i"
    return str(val)

if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1-M16 Comprehensive Features:**
    - Standard Operations, LU/Cholesky Decompositions, SVD, QR, Projections.
    - $Ax=b$ Consistency Analysis, Least-Squares, Vector Spaces & Bases.
    - Linear Mappings, Fundamental Subspaces, Pos-Def Geometries ($x^H A x$).
    - Eigenvalues, Polynomials, Diagonalization ($PDP^{-1}$), Jordan Forms.
    - **[M16]** Numerical Eigenvalue Algorithms (Power Iteration, Rayleigh Quotient), Schur Decomposition ($ZTZ^H$), Eigenvalue Condition Numbers, and Complex Spectrum Visualization.
    """)

elif choice == "Matrix Calculator":
    st.title("Standard Matrix Calculator")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "1,2; 3,4"))
    with col2: B = parse_input(st.text_area("Matrix B", "5,6; 7,8"))
    if st.button("Compute A+B"): st.latex("A+B = " + MatrixOperations.add(A, B).to_latex())

elif choice == "Advanced Factorizations Lab":
    st.title("LU, Cholesky & Advanced Factorizations Laboratory")
    A = parse_input(st.text_area("Square Matrix", "4, 12, -16; 12, 37, -43; -16, -43, 98"))
    if A and st.button("Compute LU"):
        report = LUEngine.compute_partial_pivot(A, tol)
        st.latex("P = " + report['P'].to_latex())

elif choice == "Pos-Def & Quadratic Forms":
    st.title("Definiteness, Quadratic Forms & Matrix Inequalities")
    A = parse_input(st.text_area("Hermitian Matrix A", "2, -1; -1, 2"))
    if A and st.button("Classify Matrix"):
        report = DefinitenessClassifier.classify_via_eigenvalues(A, tol)
        st.write(f"**Classification:** `{report['status']}`")

# ======================= NEW M16 NUMERICAL EIGENVALUE LAB =======================
elif choice == "Numerical Eigen & Schur Lab":
    st.title("Numerical Eigenvalue Algorithms & Schur Decomposition")
    st.markdown("Explore Iterative Eigenvalue methods, Schur Decomposition, and Eigenvalue Sensitivity.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Iterative Eigen Solvers", 
        "QR Iteration (Educational)", 
        "Schur Decomposition",
        "Eigenvalue Sensitivity",
        "Algorithm Comparison",
        "🎓 Learning Mode"
    ])

    A_str = st.text_area("Target Square Matrix $A$", "4, -1, 1; -1, 3, -2; 1, -2, 3", key="M16_A")
    A = parse_input(A_str)

    with tab1:
        st.subheader("Iterative Eigenvalue & Eigenvector Computation")
        method = st.radio("Select Method", ["Power Iteration", "Inverse Iteration", "Rayleigh Quotient Iteration"])
        x0_str = st.text_input("Initial Guess Vector $x_0$", "1, 1, 1")
        
        shift_val = 0.0
        if method == "Inverse Iteration":
            shift_val = st.number_input("Shift $\\mu$", value=2.0)
            
        max_iter = st.number_input("Max Iterations", value=100, step=10)
        
        if A and x0_str and st.button("Run Iterative Solver"):
            try:
                x0 = parse_vector_list(x0_str)[0]
                
                if method == "Power Iteration":
                    lam, vec, hist, conv = IterativeEigenEngine.power_iteration(A, x0, max_iter, tol)
                elif method == "Inverse Iteration":
                    lam, vec, hist, conv = IterativeEigenEngine.inverse_iteration(A, x0, shift_val, max_iter, tol)
                elif method == "Rayleigh Quotient Iteration":
                    lam, vec, hist, conv = IterativeEigenEngine.rayleigh_quotient_iteration(A, x0, max_iter, tol)
                
                if conv:
                    st.success(f"✅ Converged in {len(hist)} iterations.")
                else:
                    st.warning(f"⚠️ Failed to converge within {max_iter} iterations.")
                    
                st.write(f"**Estimated Eigenvalue $\\lambda$:** `{format_complex(lam)}`")
                st.latex("v \\approx [" + ', '.join([format_complex(x) for x in vec.vector]) + "]^T")
                st.write(f"**Final Residual $\\vert{}\\vert{}Av - \\lambda v\\vert{}\\vert{}_2$:** `{hist[-1]['residual']:e}`")
                
                st.plotly_chart(plot_convergence_history(hist, metric='residual'))
            except Exception as e:
                st.error(f"Iterative Solver Error: {e}")

    with tab2:
        st.subheader("Educational QR Algorithm (Unshifted)")
        st.write("Iterates $A_{k+1} = R_k Q_k$. Converges to a Schur/Diagonal form.")
        qr_iters = st.number_input("Max Iterations (QR)", value=30, step=5)
        
        if A and st.button("Run QR Iteration"):
            try:
                A_inf, hist = QREigenAlgorithm.educational_qr_iteration(A, max_iter=qr_iters, tol=tol)
                
                st.write(f"Ran {len(hist)} iterations.")
                st.markdown("#### Final Matrix $A_{\\infty}$ (Approximate Schur Form)")
                st.latex(A_inf.to_latex())
                
                st.write("**Diagonal Elements (Approximate Eigenvalues):**")
                diag_vals = np.diag(np.array(A_inf.matrix))
                st.write([format_complex(d) for d in diag_vals])
                
                st.plotly_chart(plot_convergence_history(hist, metric='off_diag'))
            except Exception as e:
                st.error(f"QR Iteration Error: {e}")

    with tab3:
        st.subheader("Schur Decomposition ($A = ZTZ^H$)")
        mode = st.radio("Output Form", ["Complex Schur (Strict Upper Triangular)", "Real Schur (Quasi-Upper Triangular)"])
        
        if A and st.button("Compute Schur Decomposition"):
            try:
                out_mode = 'real' if 'Real' in mode else 'complex'
                schur_res = SchurDecomposition.compute(A, output=out_mode)
                
                st.latex("T = " + schur_res['T'].to_latex())
                st.latex("Z = " + schur_res['Z'].to_latex())
                
                st.write(f"**Reconstruction Error $\\vert{}\\vert{}A - ZTZ^H\\vert{}\\vert{}_F$:** `{schur_res['reconstruction_error']:e}`")
                st.write(f"**Unitarity Error $\\vert{}\\vert{}Z^HZ - I\\vert{}\\vert{}_F$:** `{schur_res['unitarity_error']:e}`")
                
                st.write("**Eigenvalues Extracted from T:**")
                st.write([format_complex(e) for e in schur_res['eigenvalues']])
            except Exception as e:
                st.error(f"Schur Decomposition Error: {e}")

    with tab4:
        st.subheader("Eigenvalue Sensitivity & Perturbation")
        noise_mag = st.number_input("Perturbation Magnitude (Noise Norm)", value=1e-3, format="%.1e")
        seed = st.number_input("Random Seed (Optional)", value=42)
        
        if A and st.button("Analyze Sensitivity"):
            try:
                # Condition Numbers
                conds = EigenSensitivity.compute_condition_numbers(A)
                st.write("#### Eigenvalue Condition Numbers $\\kappa(\\lambda)$")
                for c in conds:
                    st.write(f"- $\\lambda = {format_complex(c['eigenvalue'])}$ $\\rightarrow$ $\\kappa = {c['condition_number']:.4g}$")
                
                # Perturbation Experiment
                pert = EigenSensitivity.perturbation_experiment(A, noise_mag, int(seed))
                st.plotly_chart(plot_eigenvalue_spectrum(pert['original_eigenvalues'], pert['perturbed_eigenvalues']))
                
                if edu_mode:
                    st.info("💡 Condition number $\\kappa(\\lambda) = \\frac{1}{|y^H x|}$ measures sensitivity. If left and right eigenvectors are nearly orthogonal, $\\kappa$ is huge, and tiny matrix perturbations will wildly shift the eigenvalue.")
            except Exception as e:
                st.error(f"Sensitivity Error: {e}")

    with tab5:
        st.subheader("Algorithm Comparison Laboratory")
        st.write("Compares iteration counts and residuals across methods for a fixed target vector.")
        comp_x0 = st.text_input("Shared Initial Vector", "1, 1, 1", key="comp_x")
        
        if A and st.button("Compare Methods"):
            try:
                x0 = parse_vector_list(comp_x0)[0]
                results = []
                
                # SciPy Reference
                ref_w = spla.eigvals(np.array(A.matrix, dtype=float if not np.iscomplexobj(A.matrix) else complex))
                
                # Power
                lam_p, _, hist_p, c_p = IterativeEigenEngine.power_iteration(A, x0, 100, tol)
                results.append({"Method": "Power Iteration", "Status": "Conv" if c_p else "Fail", "Iter": len(hist_p), "Lambda": lam_p, "Residual": hist_p[-1]['residual']})
                
                # RQI
                lam_r, _, hist_r, c_r = IterativeEigenEngine.rayleigh_quotient_iteration(A, x0, 100, tol)
                results.append({"Method": "Rayleigh Quotient", "Status": "Conv" if c_r else "Fail", "Iter": len(hist_r), "Lambda": lam_r, "Residual": hist_r[-1]['residual']})
                
                import pandas as pd
                st.table(pd.DataFrame(results))
                
                st.write("**Reference Spectrum (SciPy LAPACK):**")
                st.write([format_complex(w) for w in ref_w])
            except Exception as e:
                st.error(f"Comparison Error: {e}")

    with tab6:
        st.subheader("🎓 Educational Learning Mode")
        with st.expander("1. Power Iteration & Dominant Eigenvalues"):
            st.write("Repeatedly multiplying a vector by $A$ ($y = Ax$) naturally aligns the vector with the dominant eigenvector (the one corresponding to the largest $|\lambda|$). Normalizing at each step prevents overflow.")
        with st.expander("2. Inverse Iteration & Matrix Shifts"):
            st.write("By iterating $(A - \\mu I)^{-1} x$, we perform Power Iteration on the inverse shifted matrix. The largest eigenvalue of the inverse corresponds to the eigenvalue of $A$ closest to the shift $\\mu$. We NEVER explicitly invert the matrix; we solve the system $(A - \\mu I)y = x$ using fast numerical LU/QR solvers.")
        with st.expander("3. Rayleigh Quotient Iteration (RQI)"):
            st.write("RQI dynamically updates the shift $\\mu$ using the Rayleigh Quotient $\\frac{x^H A x}{x^H x}$ at every step. For symmetric matrices, this yields extremely rapid (cubic) convergence to an eigenvalue, provided the initial guess isn't exactly orthogonal to the target eigenspace.")
        with st.expander("4. Schur Decomposition ($ZTZ^H$)"):
            st.write("Not every matrix can be diagonalized ($PDP^{-1}$), but *every* square matrix can be factored into Schur form ($ZTZ^H$), where $Z$ is unitary (orthogonal) and $T$ is upper triangular. The eigenvalues sit perfectly on the diagonal of $T$. This makes Schur decomposition the bedrock of stable numerical eigenvalue algorithms.")

# ==============================================================================

elif choice == "Vector Calculator":
    st.title("3D Vector Laboratory")
    v1_str = st.text_input("Vector u", "1, 2, 3")
    v2_str = st.text_input("Vector v", "4, 5, 6")
    try:
        u = VectorEngine([float(x) for x in v1_str.split(',')])
        v = VectorEngine([float(x) for x in v2_str.split(',')])
        st.latex(f"\\mathbf{{u}} \\cdot \\mathbf{{v}} = {VectorEngine.dot_product(u, v)}")
    except Exception as e:
        st.error(f"Error: {e}")

elif choice == "Validation Lab":
    st.title("Validation Laboratory")
    A = parse_input(st.text_area("Matrix A", "4,7; 2,6", key="ValA"))
    if A and st.button("Validate $AA^{-1} = I$"):
        is_val, res_val, I_calc = ValidationEngine.verify_inverse(A, tol)
        st.write("✅ VERIFIED" if is_val else "❌ FAILED", f"(Residual: {res_val:e})")
