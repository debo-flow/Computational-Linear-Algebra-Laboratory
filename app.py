import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla

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

# ================= NEW M11 IMPORTS =================
from linear_algebra.svd.core import SVDEngine
from linear_algebra.svd.approximation import LowRankApproximation
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine
from visualization.svd_plots import plot_singular_values, plot_energy_retention
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
    "SVD & Low-Rank Approx Lab",    # NEW M11 TAB
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
        data = [[float(val) for val in row.split(',')] for row in rows]
        return MatrixEngine(data)
    except Exception as e:
        st.error(f"Input Error: Please use format '1,2; 3,4'. Detail: {e}")
        return None

def parse_vector_list(text):
    try:
        vec_strs = text.split('|')
        return [VectorEngine([float(x) for x in v.split(',')]) for v in vec_strs if v.strip()]
    except Exception:
        return None

def format_complex(val):
    if isinstance(val, (int, float, np.float64, np.float32)):
        return f"{val:.4g}"
    if isinstance(val, (complex, np.complex128)):
        if abs(val.imag) < 1e-9: return f"{val.real:.4g}"
        sign = "+" if val.imag >= 0 else "-"
        return f"{val.real:.4g} {sign} {abs(val.imag):.4g}i"
    return str(val)

if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1-M11 Comprehensive Features:**
    - Standard Operations, LU/Cholesky Decompositions.
    - $Ax=b$ Consistency Analysis & Exact Solvers.
    - Vector Spaces, Linear Mappings, Subspaces & Orthogonality.
    - Eigenvalues, Diagonalization, Similarity ($P^{-1}AP$), & Jordan Forms.
    - **[M11]** Singular Value Decomposition (SVD), Eckart-Young Rank Approximations, Pseudoinverses ($A^+$), and Matrix Compression.
    """)

elif choice == "Matrix Calculator":
    st.title("Standard Matrix Calculator")
    col1, col2 = st.columns(2)
    with col1:
        A = parse_input(st.text_area("Matrix A", "1,2; 3,4", key="A"))
        if A: st.latex("A = " + A.to_latex())
    with col2:
        B = parse_input(st.text_area("Matrix B", "5,6; 7,8", key="B"))
        if B: st.latex("B = " + B.to_latex())

    operation = st.selectbox("Operation", ["Addition (A+B)", "Multiplication (AB)", "Determinant det(A)", "Inverse A⁻¹"])
    if st.button("Compute"):
        try:
            if operation == "Addition (A+B)": st.latex("A + B = " + MatrixOperations.add(A, B).to_latex())
            elif operation == "Multiplication (AB)": st.latex("A B = " + MatrixOperations.multiply(A, B).to_latex())
            elif operation == "Determinant det(A)": st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
            elif operation == "Inverse A⁻¹": st.latex("A^{-1} = " + MatrixOperations.inverse(A, tol).to_latex())
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Advanced Operations":
    st.title("Advanced Matrix Operations")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "2,1; 1,2", key="AdvA"))
    with col2: B = parse_input(st.text_area("Matrix B (if needed)", "1,0; 0,1", key="AdvB"))
    adv_op = st.selectbox("Operation", ["Hadamard Product (A ∘ B)", "Element-wise Division (A ./ B)", "Kronecker Product (A ⊗ B)", "Matrix Power (A^n)", "Matrix Exponential exp(A)", "Scalar Shift (A + λI)"])
    if adv_op == "Matrix Power (A^n)": n_val = st.number_input("Exponent n", value=2, step=1)
    if adv_op == "Scalar Shift (A + λI)": lambda_val = st.number_input("Lambda (λ)", value=1.0)
    if st.button("Compute Advanced"):
        try:
            if adv_op == "Hadamard Product (A ∘ B)": st.latex("A \\circ B = " + AdvancedMatrixOperations.hadamard_product(A, B).to_latex())
            elif adv_op == "Element-wise Division (A ./ B)": st.latex("A ./ B = " + AdvancedMatrixOperations.elementwise_division(A, B, tol).to_latex())
            elif adv_op == "Kronecker Product (A ⊗ B)": st.latex("A \\otimes B = " + AdvancedMatrixOperations.kronecker_product(A, B).to_latex())
            elif adv_op == "Matrix Power (A^n)": st.latex(f"A^{{{n_val}}} = " + AdvancedMatrixOperations.matrix_power(A, n_val).to_latex())
            elif adv_op == "Matrix Exponential exp(A)": st.latex("e^A = " + AdvancedMatrixOperations.matrix_exponential(A).to_latex())
            elif adv_op == "Scalar Shift (A + λI)": st.latex(f"A + {lambda_val}I = " + AdvancedMatrixOperations.scalar_shift(A, lambda_val).to_latex())
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Decompositions":
    st.title("Matrix Decompositions")
    A = parse_input(st.text_area("Matrix A", "4,12,-16; 12,37,-43; -16,-43,98"))
    dec_op = st.selectbox("Decomposition", ["LU Decomposition (PA = LU)", "Cholesky (A = LL^T)"])
    if st.button("Decompose"):
        try:
            if dec_op == "LU Decomposition (PA = LU)":
                P, L, U = MatrixDecompositions.lu_decomposition(A)
                st.latex("P = " + P.to_latex() + ", L = " + L.to_latex() + ", U = " + U.to_latex())
            elif dec_op == "Cholesky (A = LL^T)":
                L = MatrixDecompositions.cholesky_decomposition(A)
                st.latex("L = " + L.to_latex())
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Determinants & Adjugate":
    st.title("Determinants, Minors & Adjugate")
    A = parse_input(st.text_area("Matrix A", "1,2,3; 0,1,4; 5,6,0"))
    if A and st.button("Compute Full Analysis"):
        try:
            st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
            st.latex("C = " + MinorsCofactors.cofactor_matrix(A).to_latex())
            st.latex("\\text{adj}(A) = " + MinorsCofactors.adjugate(A).to_latex())
        except Exception as e: st.error(str(e))

elif choice == "Rank & Singularity":
    st.title("Singularity & Invertibility")
    A = parse_input(st.text_area("Matrix A", "1,2,3; 2,4,6; 3,6,9"))
    if A and st.button("Analyze"):
        report = SingularityAnalyzer.analyze(A, tol)
        st.write(f"**Rank:** {report['rank']}, **Condition Number:** {report.get('condition_number', 'N/A')}")

elif choice == "Linear Systems Lab":
    st.title("Linear Systems Laboratory: $AX = B$")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "2,1,-1; -3,-1,2; -2,1,2"))
    with col2: B = parse_input(st.text_area("Vector B", "8; -11; -3"))
    if A and B and st.button("Solve"):
        try:
            system = LinearSystem(A, B)
            sol = SystemSolvers.parametric_solve(system)
            st.latex("X = " + sol.to_latex())
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Vector Spaces Lab":
    st.title("Vector Spaces & Spans")
    span_v_str = st.text_input("Spanning Vectors", "1,2,3 | 4,5,6")
    target_str = st.text_input("Target Vector", "7,8,9")
    if st.button("Analyze Span Membership"):
        try:
            vectors = parse_vector_list(span_v_str)
            target = parse_vector_list(target_str)[0]
            in_span, _ = SpanAnalyzer.is_in_span(target, vectors, tol)
            if in_span: st.success("✅ The target vector is INSIDE the span.")
            else: st.error("❌ The target vector is OUTSIDE the span.")
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Basis & Coordinates Lab":
    st.title("Basis Extraction")
    v_str = st.text_input("Generating Set", "1,2,3 | 2,4,6 | 1,0,1")
    if st.button("Extract Basis"):
        try:
            vectors = parse_vector_list(v_str)
            basis, pivots = BasisAnalyzer.extract_basis(vectors)
            for i, v in enumerate(basis): st.latex(f"b_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Linear Transformations Lab":
    st.title("Linear Transformations Lab")
    A = parse_input(st.text_area("Matrix A", "0,-1; 1,0"))
    if A and st.button("Plot 2D Transformation"):
        if A.shape == (2, 2) and not A.is_symbolic:
            st.plotly_chart(plot_2d_transformation(A.matrix))

elif choice == "Fundamental Subspaces Lab":
    st.title("Fundamental Subspaces")
    A = parse_input(st.text_area("Matrix A", "1,2,3; 4,5,6; 7,8,9"))
    if A and st.button("Analyze All Four Subspaces"):
        try:
            report = FundamentalSubspaces.analyze(A, tol)
            st.write(f"**Rank:** {report['rank']}, **Nullity:** {report['nullity']}, **Left Nullity:** {report['left_nullity']}")
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Eigenvalues & Polynomials":
    st.title("Eigenvalues & Eigenvectors")
    A = parse_input(st.text_area("Matrix A", "1, 2; 2, 1"))
    if A and st.button("Compute Eigen-decomposition"):
        try:
            eigen_data = EigenEngine.compute_eigen_data(A, tol)
            for i, data in enumerate(eigen_data):
                st.markdown(f"### $\\lambda_{i+1} = {format_complex(data['eigenvalue'])}$")
                st.write(f"- **AM:** {data['algebraic_multiplicity']}, **GM:** {data['geometric_multiplicity']}")
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Matrix Diagonalization Lab":
    st.title("Matrix Diagonalization ($A=PDP^{-1}$)")
    A = parse_input(st.text_area("Square Matrix A", "4, -2; 1, 1"))
    if A and st.button("Diagonalize Matrix"):
        try:
            report = DiagonalizationEngine.diagonalize(A, tol)
            if report['is_diagonalizable']:
                st.latex("P = " + report['P'].to_latex())
                st.latex("D = " + report['D'].to_latex())
            else: st.error(report['message'])
        except Exception as e: st.error(f"Error: {e}")

# ======================= NEW M11 SVD & LOW-RANK APPROX LAB =======================
elif choice == "SVD & Low-Rank Approx Lab":
    st.title("SVD, Low-Rank Approx & Pseudoinverse Lab")
    st.markdown("Analyze Singular Values ($A = U\\Sigma V^H$), matrix compression, and overdetermined Least-Squares.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "SVD Calculator", 
        "Low-Rank Approximation", 
        "Pseudoinverse ($A^+$)",
        "SVD Least-Squares",
        "🎓 Learning Mode"
    ])

    A_str = st.text_area("Matrix $A_{m \\times n}$", "1,2; 3,4; 5,6")
    A = parse_input(A_str)

    with tab1:
        st.subheader("Singular Value Decomposition $A = U \\Sigma V^H$")
        svd_mode = st.radio("SVD Mode", ["Full SVD", "Reduced (Economy) SVD"])
        
        if A and st.button("Compute SVD"):
            try:
                is_full = (svd_mode == "Full SVD")
                svd_data = SVDEngine.compute(A, full_matrices=is_full, tol=tol)
                
                c1, c2, c3 = st.columns(3)
                with c1: st.write(f"- **Num Rank:** {svd_data['rank']}")
                with c2: st.write(f"- **Condition Number $\\kappa_2(A)$:** `{svd_data['condition_number']:.2e}`")
                with c3: st.write(f"- **Max $\\sigma$:** `{svd_data['max_sv']:.4g}`")
                
                st.latex("U = " + svd_data['U'].to_latex())
                st.write(f"Singular Values $\\Sigma$: `[{', '.join([f'{s:.4g}' for s in svd_data['S']])}]`")
                st.latex("V^H = " + svd_data['Vh'].to_latex())
                
                A_recon = SVDEngine.reconstruct(svd_data['U'], svd_data['S'], svd_data['Vh'])
                recon_err = np.linalg.norm(np.array(A.matrix, dtype=float) - np.array(A_recon.matrix, dtype=float), 'fro')
                st.write(f"Reconstruction Error $\\vert{}\\vert{}A - U\\Sigma V^H\\vert{}\\vert{}_F$: `{recon_err:e}`")
                
                st.plotly_chart(plot_singular_values(svd_data['S'], log_scale=True))
                
                if svd_data['condition_number'] > 1e10:
                    st.warning("⚠️ Matrix is ill-conditioned or computationally singular. Expect extreme numerical sensitivity to perturbations.")
                    
            except Exception as e:
                st.error(f"SVD Error: {e}")

    with tab2:
        st.subheader("Eckart-Young Truncated SVD (Compression)")
        if A:
            k = st.number_input("Truncation Rank $k$", min_value=1, max_value=min(A.rows, A.cols), value=1, step=1)
            if st.button("Compute Rank-k Approximation"):
                try:
                    approx = LowRankApproximation.truncate(A, k, tol)
                    
                    st.latex(f"A_{{{k}}} \\approx " + approx['A_k'].to_latex())
                    
                    st.write("#### Approximation Errors")
                    st.write(f"- **Computed Absolute Error $\\vert{}\\vert{}A - A_k\\vert{}\\vert{}_F$:** `{approx['abs_error_frob']:.4g}`")
                    st.write(f"- **Theoretical Eckart-Young Error $\\sqrt{{\\sum_{{i=k+1}}^r \\sigma_i^2}}$:** `{approx['theoretical_frob_error']:.4g}`")
                    st.write(f"- **Spectral Error $\\sigma_{{{k+1}}}$:** `{approx['theoretical_spectral_error']:.4g}`")
                    st.write(f"- **Relative Frob Error:** `{approx['rel_error_frob']:.2%}`")
                    
                    st.plotly_chart(plot_energy_retention(SVDEngine.compute(A, full_matrices=False)['S']))
                except Exception as e:
                    st.error(f"Approximation Error: {e}")

    with tab3:
        st.subheader("Moore-Penrose Pseudoinverse $A^+$")
        st.write("Verifies the four defining conditions of the pseudoinverse.")
        if A and st.button("Compute $A^+$"):
            try:
                A_plus = PseudoinverseEngine.compute(A, tol)
                st.latex("A^+ = " + A_plus.to_latex())
                
                checks = PseudoinverseEngine.verify_mp_conditions(A, A_plus, tol)
                st.markdown("#### Moore-Penrose Conditions Verification")
                st.write(f"1. $A A^+ A = A$ $\\rightarrow$ Residual: `{checks['cond1_err']:e}` {'✅' if checks['cond1_valid'] else '❌'}")
                st.write(f"2. $A^+ A A^+ = A^+$ $\\rightarrow$ Residual: `{checks['cond2_err']:e}` {'✅' if checks['cond2_valid'] else '❌'}")
                st.write(f"3. $(A A^+)^H = A A^+$ $\\rightarrow$ Residual: `{checks['cond3_err']:e}` {'✅' if checks['cond3_valid'] else '❌'}")
                st.write(f"4. $(A^+ A)^H = A^+ A$ $\\rightarrow$ Residual: `{checks['cond4_err']:e}` {'✅' if checks['cond4_valid'] else '❌'}")
                
            except Exception as e:
                st.error(f"Pseudoinverse Error: {e}")

    with tab4:
        st.subheader("SVD Least Squares (Minimum-Norm Solution)")
        st.write("Solves $Ax \\approx b$ using $x = A^+b$ for over/under-determined systems.")
        b_str = st.text_area("Target Vector b", "1; 2; 3")
        b = parse_input(b_str)
        if A and b and st.button("Compute Minimum-Norm Solution"):
            try:
                x, res_norm, sol_norm = PseudoinverseEngine.svd_least_squares(A, b, tol)
                st.latex("x_{LSTSQ} = " + x.to_latex())
                st.write(f"**Residual Norm $\\vert{}\\vert{}Ax - b\\vert{}\\vert{}_2$:** `{res_norm:.4g}`")
                st.write(f"**Solution Norm $\\vert{}\\vert{}x\\vert{}\\vert{}_2$:** `{sol_norm:.4g}`")
                
                if edu_mode:
                    st.info("💡 The pseudoinverse guarantees two things simultaneously: It finds the $x$ that minimizes the residual error $\vert{}\vert{}Ax-b\vert{}\vert{}$, and among all possible least-squares solutions, it uniquely selects the one with the smallest Euclidean length $\vert{}\vert{}x\vert{}\vert{}$.")
            except Exception as e:
                st.error(f"Least Squares Error: {e}")

    with tab5:
        st.subheader("🎓 Educational Learning Mode: SVD")
        with st.expander("1. What is SVD? ($A = U\\Sigma V^H$)"):
            st.write("SVD proves that *any* linear transformation can be broken down into three geometric steps: 1) A rotation/reflection in the domain ($V^H$), 2) independent scalings along coordinate axes ($\\Sigma$), and 3) a final rotation/reflection in the codomain ($U$).")
        with st.expander("2. Eckart-Young Theorem (Matrix Compression)"):
            st.write("If you want to approximate a matrix $A$ using a lower rank matrix $A_k$, throwing away the smallest singular values in SVD gives you the mathematically optimal approximation. The error is exactly the magnitude of the largest singular value you threw away.")
        with st.expander("3. Condition Number $\\kappa_2(A)$"):
            st.write("$\\kappa_2(A) = \\sigma_{max} / \\sigma_{min}$. A high condition number means the matrix severely distorts space, turning spheres into extremely long, thin ellipses. Computationally, this indicates extreme sensitivity to numerical rounding errors.")

# ==============================================================================
elif choice == "Vector Calculator":
    st.title("3D Vector Laboratory")
    pass # Skipped for brevity

elif choice == "Validation Lab":
    st.title("Validation Laboratory")
    pass # Skipped for brevity
