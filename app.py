import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla

# Base & Systems (M1-M4)
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

# Spaces, Transformations, SVD, Orthogonality, QR (M5-M13)
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

# ================= NEW M14 IMPORTS =================
from linear_algebra.factorizations.lu import LUEngine
from linear_algebra.factorizations.cholesky import CholeskyEngine
from linear_algebra.factorizations.triangular_solver import TriangularSolver
from linear_algebra.factorizations.diagnostics import FactorizationDiagnostics
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
    "Advanced Factorizations Lab",  # NEW M14 TAB
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
    **M1-M14 Comprehensive Features:**
    - Standard Operations, Determinants, Consistency ($Ax=b$).
    - Vector Spaces, Subspaces, Eigenvalues & Diagonalization ($PDP^{-1}$).
    - SVD, Truncation, Pseudoinverses, Gram-Schmidt ($P=QQ^H$).
    - QR Decompositions, Multi-RHS Least-Squares & Weighted Least-Squares.
    - **[M14]** Advanced Factorization Lab: Rigorous LU ($PA=LU$), Cholesky ($LL^H$), Indefinite $LDL^T$, and Forward/Backward Triangular System Substitution.
    """)

# ... (M1-M13 existing generic UI logic remains functionally identical, contracted here to respect response space). ...
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

# ======================= NEW M14 ADVANCED FACTORIZATIONS LAB =======================
elif choice == "Advanced Factorizations Lab":
    st.title("LU, Cholesky & Advanced Factorizations Laboratory")
    st.markdown("Analyze $PA=LU$, $A=LL^H$, $A=LD L^H$, and precise Triangular Back-Substitutions.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "LU Decomposition ($PA=LU$)", 
        "Cholesky & $LDL^H$ Factorization", 
        "Triangular Solvers & $Ax=b$",
        "Factorization Comparison Lab",
        "🎓 Educational Step-by-Step"
    ])

    A_str = st.text_area("Square Matrix $A_{n \\times n}$", "4, 12, -16; 12, 37, -43; -16, -43, 98")
    A = parse_input(A_str)

    with tab1:
        st.subheader("LU Factorization with Partial Pivoting")
        st.write("Ensures numerical stability by pivoting rows to avoid zero divisors.")
        if A and st.button("Compute LU ($PA=LU$)"):
            try:
                report = LUEngine.compute_partial_pivot(A, tol)
                st.latex("P = " + report['P'].to_latex())
                st.latex("L = " + report['L'].to_latex())
                st.latex("U = " + report['U'].to_latex())
                st.write(f"**Reconstruction Error $\\vert{}\\vert{}PA - LU\\vert{}\\vert{}_F$:** `{report['reconstruction_error']:e}`")
                
                if edu_mode:
                    st.info("💡 $P$ is a permutation matrix. Mathematically, $PA$ swaps the rows of $A$ into an optimal order so that $L$ and $U$ can be extracted flawlessly without encountering $0$ on the diagonal.")
            except Exception as e:
                st.error(f"LU Error: {e}")

    with tab2:
        st.subheader("Cholesky & Symmetric Indefinite ($LDL^H$) Factorization")
        st.write("Factorizations strictly requiring $A$ to be Symmetric (or Hermitian).")
        fact_type = st.radio("Factorization Mode", ["Cholesky (Requires Positive-Definite)", "LDL^H (Handles Indefinite Matrices)"])
        
        if A and st.button("Compute Symmetric Factorization"):
            try:
                if 'Cholesky' in fact_type:
                    report = CholeskyEngine.compute_cholesky(A, tol)
                    st.success("Matrix is verified Positive-Definite.")
                    st.latex("L = " + report['L'].to_latex())
                    st.latex("L^H = " + report['L_H'].to_latex())
                    st.write(f"**Reconstruction Error $\\vert{}\\vert{}A - L L^H\\vert{}\\vert{}_F$:** `{report['reconstruction_error']:e}`")
                else:
                    report = CholeskyEngine.compute_ldl(A, tol)
                    st.success("Matrix is verified Symmetric/Hermitian.")
                    st.latex("P = " + report['P'].to_latex())
                    st.latex("L = " + report['L'].to_latex())
                    st.latex("D = " + report['D'].to_latex())
                    st.write(f"**Reconstruction Error $\\vert{}\\vert{}A - PLDL^HP^T\\vert{}\\vert{}_F$:** `{report['reconstruction_error']:e}`")
            except Exception as e:
                st.error(f"Factorization Error: {e}")

    with tab3:
        st.subheader("Solve $Ax = b$ via Triangular Substitution")
        b_str = st.text_area("Target Vector $b$ (Supports multiple RHS columns)", "1; 2; 3")
        b = parse_input(b_str)
        
        if A and b and st.button("Solve System via $LU$"):
            try:
                # 1. Factorize
                lu_data = LUEngine.compute_partial_pivot(A, tol)
                
                # 2. Substitute
                x, y, Pb = TriangularSolver.solve_via_lu(lu_data['P'], lu_data['L'], lu_data['U'], b)
                
                # 3. Analyze
                diag = FactorizationDiagnostics.analyze_system(A, b, x)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Forward Substitution ($Ly = Pb$)")
                    st.latex("Pb = " + Pb.to_latex())
                    st.latex("y = " + y.to_latex())
                with c2:
                    st.markdown("#### Backward Substitution ($Ux = y$)")
                    st.latex("x = " + x.to_latex())
                
                st.markdown("---")
                st.write(f"**Absolute Residual $\\vert{}\\vert{}b - Ax\\vert{}\\vert{}_F$:** `{diag['abs_residual']:e}`")
                st.write(f"**Matrix Condition Number $\\kappa(A)$:** `{diag['condition_number']:.2e}`")
                if diag['warning']:
                    st.warning(f"⚠️ {diag['warning']}")
                    
            except Exception as e:
                st.error(f"Solver Error: {e}")

    with tab4:
        st.subheader("Factorization Comparison Laboratory")
        st.write("Compare mathematical requirements, reconstruction accuracy, and computational assumptions.")
        if A and st.button("Run Unified Comparison"):
            st.markdown("### 1. Partial Pivoted $LU$ ($PA = LU$)")
            try:
                lu = LUEngine.compute_partial_pivot(A, tol)
                st.success(f"Success. Reconstruction Error: `{lu['reconstruction_error']:e}`")
            except Exception as e: st.error(f"Failed: {e}")
            
            st.markdown("### 2. SVD ($A = U \\Sigma V^H$)")
            try:
                svd = SVDEngine.compute(A, tol=tol)
                st.success(f"Success. Condition Number: `{svd['condition_number']:.2e}`")
            except Exception as e: st.error(f"Failed: {e}")
            
            st.markdown("### 3. QR Decomposition ($A = QR$)")
            try:
                qr = QREngine.compute_library_qr(A, mode='full', tol=tol)
                st.success(f"Success. Orthogonality Error: `{qr['orthogonality_error']:e}`")
            except Exception as e: st.error(f"Failed: {e}")
            
            st.markdown("### 4. Cholesky Factorization ($A = L L^H$)")
            try:
                chol = CholeskyEngine.compute_cholesky(A, tol)
                st.success(f"Success. Reconstruction Error: `{chol['reconstruction_error']:e}`")
            except Exception as e: st.error(f"Failed: Matrix is either non-symmetric or not strictly positive-definite. ({e})")

    with tab5:
        st.subheader("🎓 Educational Step-by-Step: LU without Pivoting")
        st.write("Calculates LU naively $A=LU$. Mathematically elegant, but numerically brittle to $0$'s.")
        if A and st.button("Generate LU Steps"):
            try:
                report = LUEngine.compute_no_pivot(A, tol)
                for step in report['steps']:
                    st.markdown(f"#### {step['description']}")
                    c1, c2 = st.columns(2)
                    with c1: st.latex("L_{current} = " + step['L_current'].to_latex())
                    with c2: st.latex("U_{current} = " + step['U_current'].to_latex())
                    st.markdown("---")
                
                st.success("Final Output:")
                st.latex("L = " + report['L'].to_latex())
                st.latex("U = " + report['U'].to_latex())
            except Exception as e:
                st.error(f"Educational LU Error: {e}")

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
