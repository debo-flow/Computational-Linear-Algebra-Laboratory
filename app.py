import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla

# Milestones 1-12 Engines (Preserved completely)
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.matrix.validation import ValidationEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix.advanced_operations import AdvancedMatrixOperations
from linear_algebra.matrix.decompositions import MatrixDecompositions
from linear_algebra.matrix.analysis import MatrixAnalysis
from linear_algebra.matrix.minors_cofactors import MinorsCofactors
from linear_algebra.matrix.singularity_analysis import SingularityAnalyzer
from linear_algebra.matrix.determinant_properties import DeterminantProperties
from visualization.vector_plots import plot_3d_vectors
from visualization.matrix_plots import plot_matrix_heatmap
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.elimination import EliminationEngine
from linear_algebra.systems.verification import SystemVerification
from linear_algebra.vector_spaces.linear_combinations import LinearCombinations
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.independence import LinearIndependenceAnalyzer
from linear_algebra.vector_spaces.subspace_analysis import SubspaceAnalyzer
from linear_algebra.vector_spaces.axiom_checker import AxiomExplorer
from linear_algebra.vector_spaces.basis import BasisAnalyzer
from linear_algebra.vector_spaces.dimension import DimensionAnalyzer
from linear_algebra.vector_spaces.coordinates import CoordinateSystem
from linear_algebra.transformations.geometric import GeometricTransformations
from linear_algebra.transformations.analysis import TransformationAnalyzer
from linear_algebra.transformations.basis_representation import TransformationBasis
from linear_algebra.transformations.composition import TransformationComposition
from visualization.transformation_plots import plot_2d_transformation
from linear_algebra.subspaces.fundamental import FundamentalSubspaces
from linear_algebra.subspaces.orthogonality import SubspaceOrthogonality
from linear_algebra.subspaces.comparison import SubspaceComparison
from linear_algebra.eigenvalues.core import EigenEngine
from linear_algebra.eigenvalues.characteristic import CharacteristicPolynomial
from linear_algebra.eigenvalues.diagnostics import EigenDiagnostics
from linear_algebra.diagonalization.core import DiagonalizationEngine
from linear_algebra.diagonalization.similarity import SimilarityAnalyzer
from linear_algebra.diagonalization.functions import MatrixFunctions
from linear_algebra.diagonalization.jordan import JordanCanonicalForm
from linear_algebra.svd.core import SVDEngine
from linear_algebra.svd.approximation import LowRankApproximation
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine
from visualization.svd_plots import plot_singular_values, plot_energy_retention
from linear_algebra.orthogonality.inner_product import InnerProductEngine
from linear_algebra.orthogonality.gram_schmidt import GramSchmidtEngine
from linear_algebra.orthogonality.projections import ProjectionEngine

# ================= NEW M13 IMPORTS =================
from linear_algebra.qr.core import QREngine
from linear_algebra.qr.least_squares import QRLeastSquares
from linear_algebra.qr.educational import QREducational
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
    "QR & Least-Squares Lab",       # NEW M13 TAB
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
    **M1-M13 Comprehensive Features:**
    - Standard Operations, Determinants, Consistency ($Ax=b$).
    - Spaces, Bases, Subspaces, Subspace Orthogonality.
    - Linear Transformations, Eigenvalues, Diagonalization ($PDP^{-1}$), & Jordan Forms.
    - SVD, Truncation, Pseudoinverses, Gram-Schmidt ($P=QQ^H$).
    - **[M13]** QR Decompositions (Householder, Givens, Pivoted), Multi-RHS Least-Squares ($Rx = Q^Hb$), and Weighted Least-Squares.
    """)

# ... (M1-M12 Unchanged UI Logic Blocks Omitted for brevity in this print out, 
# but they reside here exactly as generated in M12). ...

elif choice == "Matrix Calculator":
    st.title("Standard Matrix Calculator")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "1,2; 3,4", key="A"))
    with col2: B = parse_input(st.text_area("Matrix B", "5,6; 7,8", key="B"))
    op = st.selectbox("Operation", ["Addition (A+B)", "Multiplication (AB)", "Determinant det(A)", "Inverse A⁻¹"])
    if st.button("Compute"):
        try:
            if op == "Addition (A+B)": st.latex("A+B = " + MatrixOperations.add(A, B).to_latex())
            elif op == "Multiplication (AB)": st.latex("AB = " + MatrixOperations.multiply(A, B).to_latex())
            elif op == "Determinant det(A)": st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
            elif op == "Inverse A⁻¹": st.latex("A^{-1} = " + MatrixOperations.inverse(A, tol).to_latex())
        except Exception as e: st.error(f"Error: {e}")

# ======================= NEW M13 QR & LEAST-SQUARES LAB =======================
elif choice == "QR & Least-Squares Lab":
    st.title("QR Decomposition & Least-Squares Laboratory")
    st.markdown("Analyze $A=QR$ Factorizations, Householder/Givens Eliminations, and robust Least-Squares optimization.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "QR Factorization Modes", 
        "Pivoted Rank-Revealing QR", 
        "Householder/Givens Steps",
        "QR Least-Squares",
        "Weighted Least-Squares"
    ])

    A_str = st.text_area("Matrix $A_{m \\times n}$", "1, -1, 4; 1, 4, -2; 1, 4, 2; 1, -1, 0")
    A = parse_input(A_str)

    with tab1:
        st.subheader("Standard QR Factorization $A = QR$")
        mode = st.radio("QR Algorithm", ["LAPACK Reduced (Economy)", "LAPACK Full", "Modified Gram-Schmidt", "Classical Gram-Schmidt"])
        
        if A and st.button("Compute QR"):
            try:
                if 'LAPACK' in mode:
                    qr_data = QREngine.compute_library_qr(A, mode='reduced' if 'Reduced' in mode else 'full', tol=tol)
                else:
                    gs_method = 'modified' if 'Modified' in mode else 'classical'
                    qr_data = QREngine.compute_gram_schmidt_qr(A, method=gs_method, tol=tol)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.latex("Q = " + qr_data['Q'].to_latex())
                    st.write(f"**Orthogonality Error $\\vert{}\\vert{}Q^H Q - I\\vert{}\\vert{}_F$:** `{qr_data['orthogonality_error']:e}`")
                with c2:
                    st.latex("R = " + qr_data['R'].to_latex())
                    st.write(f"**Reconstruction Error $\\vert{}\\vert{}A - QR\\vert{}\\vert{}_F$:** `{qr_data['reconstruction_error']:e}`")
                    
                if 'Gram-Schmidt' in mode and qr_data['orthogonality_error'] > 1e-5:
                    st.warning("⚠️ High orthogonality loss detected. Gram-Schmidt is numerically unstable for ill-conditioned matrices compared to LAPACK (Householder).")
            except Exception as e:
                st.error(f"QR Error: {e}")

    with tab2:
        st.subheader("Rank-Revealing Column-Pivoted QR ($AP = QR$)")
        if A and st.button("Compute Pivoted QR"):
            try:
                pqr = QREngine.compute_pivoted_qr(A, tol=tol)
                st.write(f"**Estimated Numerical Rank:** {pqr['rank']}")
                st.write(f"**Permutation Vector $P$ (0-indexed):** `{pqr['P']}`")
                
                st.latex("Q = " + pqr['Q'].to_latex())
                st.latex("R = " + pqr['R'].to_latex())
                
                st.write(f"**Diagonal Elements of R ($\vert{}R_{{ii}}\vert{}$):** `[{', '.join([f'{x:.4e}' for x in pqr['diag_R']])}]`")
                if edu_mode:
                    st.info("💡 Pivoted QR automatically pushes the most linearly independent columns to the front. The diagonal entries of $R$ monotonically decrease, allowing us to accurately estimate the rank by looking for a sharp drop-off near the tolerance.")
            except Exception as e:
                st.error(f"Pivoted QR Error: {e}")

    with tab3:
        st.subheader("Educational Step-by-Step Transformations")
        st.write("Visualizes the step-by-step zeroing of subdiagonal elements to form upper-triangular $R$.")
        edu_method = st.radio("Transformation Type", ["Householder Reflections", "Givens Rotations"])
        
        if A and st.button("Generate Transformation Steps"):
            try:
                if edu_method == "Householder Reflections":
                    Q, R, steps = QREngine.householder_qr(A)
                    for step in steps:
                        st.markdown(f"#### Step {step['step']}: Apply $H_{step['step']}$")
                        c1, c2 = st.columns(2)
                        with c1: st.latex("H_{step} = " + step['H'].to_latex())
                        with c2: st.latex("R_{current} = " + step['R_current'].to_latex())
                        st.markdown("---")
                else:
                    Q, R, steps = QREngine.givens_qr(A)
                    for step in steps:
                        st.markdown(f"#### Step {step['step']}: {step['target']}")
                        c1, c2 = st.columns(2)
                        with c1: st.latex("G = " + step['G'].to_latex())
                        with c2: st.latex("R_{current} = " + step['R_current'].to_latex())
                        st.markdown("---")
                
                st.success("Transformation Complete.")
                st.latex("Q_{final} = " + Q.to_latex())
                st.latex("R_{final} = " + R.to_latex())
            except Exception as e:
                st.error(f"Educational QR Error: {e}")

    with tab4:
        st.subheader("QR-Based Least Squares (Multiple RHS)")
        st.write("Solves $Ax \\approx B$ robustly using $Rx = Q^H B$.")
        B_str = st.text_area("Target Matrix $B$ (Supports multiple columns)", "3; 7; 2; 1")
        B = parse_input(B_str)
        
        if A and B and st.button("Solve via QR"):
            try:
                res = QRLeastSquares.solve(A, B, tol)
                if res['status'] == "rank_deficient":
                    st.error(f"Matrix is rank-deficient (Rank = {res['rank']} < {A.cols}). Standard QR back-substitution fails. Please use the SVD Pseudoinverse Solver (M11) for the minimum-norm solution.")
                else:
                    st.latex("X_{LSTSQ} = " + res['X'].to_latex())
                    st.write(f"**Overall Residual $\\vert{}\\vert{}AX - B\\vert{}\\vert{}_F$:** `{res['overall_residual']:e}`")
                    for i, norm in enumerate(res['residual_norms']):
                        st.write(f"- RHS Column {i+1} Residual $\\Vert{}r\\Vert{}_2$: `{norm:.4g}`")
            except Exception as e:
                st.error(f"QR Least Squares Error: {e}")

    with tab5:
        st.subheader("Weighted Least Squares $\\min \\, (Ax-b)^H W (Ax-b)$")
        B_str_w = st.text_area("Target Vector $b$", "3; 7; 2; 1", key="w_b")
        W_str = st.text_input("Diagonal Weights (comma separated, positive)", "1, 100, 1, 1")
        
        if A and B_str_w and st.button("Solve Weighted System"):
            try:
                B_w = parse_input(B_str_w)
                weights = [float(w) for w in W_str.split(',')]
                
                res = QRLeastSquares.solve_weighted(A, B_w, weights, tol)
                if res['status'] == "rank_deficient":
                    st.error("Transformed Matrix is rank-deficient.")
                else:
                    st.latex("x_{WLS} = " + res['X'].to_latex())
                    if edu_mode:
                        st.info("💡 By applying heavy weights (e.g., 100) to certain equations, the least-squares solver is forced to prioritize fitting those points perfectly at the expense of ignoring errors in the lightly weighted equations.")
            except Exception as e:
                st.error(f"Weighted LS Error: {e}")

# ==============================================================================

elif choice == "Validation Lab":
    st.title("Validation Laboratory")
    A = parse_input(st.text_area("Matrix A", "4,7; 2,6", key="ValA"))
    if A and st.button("Validate $AA^{-1} = I$"):
        is_val, res_val, I_calc = ValidationEngine.verify_inverse(A, tol)
        st.write("✅ VERIFIED" if is_val else "❌ FAILED", f"(Residual: {res_val:e})")
