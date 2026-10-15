import streamlit as st
import numpy as np
import sympy as sp

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

# ================= NEW M9 IMPORTS =================
from linear_algebra.eigenvalues.core import EigenEngine
from linear_algebra.eigenvalues.characteristic import CharacteristicPolynomial
from linear_algebra.eigenvalues.diagnostics import EigenDiagnostics
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
    "Eigenvalues & Polynomials",    # NEW M9 TAB
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
    return str(val) # For SymPy

if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1-M9 Comprehensive Features:**
    - Standard Operations, LU/Cholesky Decompositions.
    - $Ax=b$ Consistency Analysis & Exact Solvers.
    - Vector Spaces (Spans, Bases, Coordinates).
    - Linear Mappings $T(x) = Ax$, Injectivity/Surjectivity.
    - The Four Fundamental Subspaces & Orthogonality.
    - **[M9]** Eigenvalues, Eigenvectors, Characteristic Polynomials, Algebraic/Geometric Multiplicity, and Spectral Identities.
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
    with col1:
        A = parse_input(st.text_area("Matrix A", "2,1; 1,2", key="AdvA"))
        if A: st.latex("A = " + A.to_latex())
    with col2:
        B = parse_input(st.text_area("Matrix B (if needed)", "1,0; 0,1", key="AdvB"))
        if B: st.latex("B = " + B.to_latex())
        
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
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Decompositions":
    st.title("Matrix Decompositions")
    A = parse_input(st.text_area("Matrix A", "4,12,-16; 12,37,-43; -16,-43,98"))
    if A: st.latex("A = " + A.to_latex())
    dec_op = st.selectbox("Decomposition", ["LU Decomposition (PA = LU)", "Cholesky (A = LL^T)"])
    if st.button("Decompose"):
        try:
            if dec_op == "LU Decomposition (PA = LU)":
                P, L, U = MatrixDecompositions.lu_decomposition(A)
                st.latex("P = " + P.to_latex() + ", L = " + L.to_latex() + ", U = " + U.to_latex())
            elif dec_op == "Cholesky (A = LL^T)":
                L = MatrixDecompositions.cholesky_decomposition(A)
                st.latex("L = " + L.to_latex())
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Determinants & Adjugate":
    st.title("Determinants, Minors & Adjugate Laboratory")
    A = parse_input(st.text_area("Matrix A", "1,2,3; 0,1,4; 5,6,0"))
    if A: 
        st.latex("A = " + A.to_latex())
        if st.button("Compute Full Analysis"):
            try:
                st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
                st.latex("C = " + MinorsCofactors.cofactor_matrix(A).to_latex())
                st.latex("\\text{adj}(A) = " + MinorsCofactors.adjugate(A).to_latex())
            except Exception as e:
                st.error(str(e))

elif choice == "Rank & Singularity":
    st.title("Singularity & Invertibility Analyzer")
    A = parse_input(st.text_area("Matrix A", "1,2,3; 2,4,6; 3,6,9"))
    if A and st.button("Analyze"):
        report = SingularityAnalyzer.analyze(A, tol)
        st.subheader(f"Status: {report['status']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- **Dimensions:** {report['dimensions']}")
            st.write(f"- **Rank:** {report['rank']}")
        with col2:
            if 'singular_values' in report: st.write(f"- **Singular Values:** {np.round(report['singular_values'], 6)}")
            if 'condition_number' in report: st.write(f"- **Condition Number $\\kappa(A)$:** {report['condition_number']:.6e}")

elif choice == "Linear Systems Lab":
    st.title("Advanced Linear Systems Laboratory: $AX = B$")
    col1, col2 = st.columns([2, 1])
    with col1:
        A_str = st.text_area("Coefficient Matrix A", "2,1,-1; -3,-1,2; -2,1,2")
        A = parse_input(A_str)
    with col2:
        B_str = st.text_area("Constants Vector B (Supports multiple columns)", "8; -11; -3")
        B = parse_input(B_str)
        
    if A and B:
        try:
            system = LinearSystem(A, B)
            st.latex(f"A = {A.to_latex()}, \\quad B = {B.to_latex()}")
            report = SystemClassifier.classify(system, tol)
            st.write(f"**Classification:** {report['status']}")
            
            if report['solution_type'] != "none" and st.button("Solve"):
                sol = SystemSolvers.parametric_solve(system)
                st.latex("X = " + sol.to_latex())
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Vector Spaces Lab":
    st.title("Vector Spaces, Span & Subspaces")
    span_v_str = st.text_input("Spanning Vectors (separated by | )", "1,2,3 | 4,5,6")
    target_str = st.text_input("Target Vector", "7,8,9")
    if st.button("Analyze Span Membership"):
        try:
            vectors = parse_vector_list(span_v_str)
            target = parse_vector_list(target_str)[0]
            in_span, report = SpanAnalyzer.is_in_span(target, vectors, tol)
            if in_span: st.success("✅ The target vector is **INSIDE** the span.")
            else: st.error("❌ The target vector is **OUTSIDE** the span.")
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Basis & Coordinates Lab":
    st.title("Basis, Coordinates & Dimension")
    v_str = st.text_input("Generating Set (separated by | )", "1,2,3 | 2,4,6 | 1,0,1")
    if st.button("Extract Basis"):
        try:
            vectors = parse_vector_list(v_str)
            basis, pivots = BasisAnalyzer.extract_basis(vectors)
            for i, v in enumerate(basis):
                st.latex(f"b_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Linear Transformations Lab":
    st.title("Linear Transformations Lab")
    A = parse_input(st.text_area("Matrix A", "0,-1; 1,0", key="cusA"))
    if A and st.button("Plot 2D Transformation"):
        st.latex("T(x) = Ax = " + A.to_latex() + " x")
        if A.shape == (2, 2) and not A.is_symbolic:
            st.plotly_chart(plot_2d_transformation(A.matrix))

elif choice == "Fundamental Subspaces Lab":
    st.title("The Four Fundamental Subspaces")
    A_str = st.text_area("Target Matrix $A_{m \\times n}$", "1,2,3; 4,5,6; 7,8,9")
    A = parse_input(A_str)
    if A and st.button("Analyze All Four Subspaces"):
        try:
            report = FundamentalSubspaces.analyze(A, tol)
            st.latex(f"A \\in \\mathbb{{R}}^{{{report['dimensions'][0]} \\times {report['dimensions'][1]}}}")
            st.write(f"**Rank:** {report['rank']}, **Nullity:** {report['nullity']}, **Left Nullity:** {report['left_nullity']}")
        except Exception as e:
            st.error(f"Error: {e}")

# ======================= NEW M9 EIGENVALUES & POLYNOMIALS LAB =======================
elif choice == "Eigenvalues & Polynomials":
    st.title("Eigenvalues, Eigenvectors & Characteristic Polynomial")
    st.markdown("Analyze spectral properties, multiplicities, eigenspaces, and diagonalizability conditions.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Eigenpairs & Eigenspaces", 
        "Characteristic Polynomial", 
        "Trace, Det & Diagonalizability",
        "🎓 Learning Mode"
    ])

    A_str = st.text_area("Square Matrix $A$", "1, 2; 2, 1")
    A = parse_input(A_str)

    with tab1:
        st.subheader("Eigenvalues ($\\lambda$) & Eigenvectors ($v$)")
        if A and st.button("Compute Eigen-decomposition"):
            try:
                eigen_data = EigenEngine.compute_eigen_data(A, tol)
                residuals = EigenDiagnostics.compute_residuals(A, eigen_data)
                
                res_idx = 0
                for i, data in enumerate(eigen_data):
                    st.markdown(f"### $\\lambda_{i+1} = {format_complex(data['eigenvalue'])}$")
                    st.write(f"- **Algebraic Multiplicity (AM):** {data['algebraic_multiplicity']}")
                    st.write(f"- **Geometric Multiplicity (GM):** {data['geometric_multiplicity']}")
                    
                    if data['is_defective']:
                        st.warning("⚠️ Defective Eigenvalue: GM < AM. Does not possess a complete basis of eigenvectors.")
                        
                    st.write("**Eigenspace Basis ($E_\\lambda$):**")
                    for j, v in enumerate(data['eigenvectors']):
                        vec_str = ', '.join([format_complex(x) for x in v.vector])
                        st.latex(f"v_{{{j+1}}} = [{vec_str}]^T")
                        if not A.is_symbolic:
                            st.write(f"Residual $\\vert{}\\vert{}Av - \\lambda v\\vert{}\\vert{}$: `{residuals[res_idx]:e}`")
                        res_idx += 1
                        
            except Exception as e:
                st.error(f"Calculation Error: {e}")

    with tab2:
        st.subheader("Characteristic Polynomial $p(\\lambda) = \\det(\\lambda I - A)$")
        if A and st.button("Compute Polynomial"):
            try:
                expr, roots, is_sym = CharacteristicPolynomial.compute(A)
                
                if is_sym:
                    st.latex(f"p(\\lambda) = {sp.latex(expr)}")
                    st.write("**Exact Roots:**")
                    for r in roots:
                        st.latex(f"\\lambda = {sp.latex(r)}")
                else:
                    poly_str = CharacteristicPolynomial.format_numeric_poly(expr)
                    st.latex(f"p(\\lambda) = {poly_str}")
                    st.write("**Numerical Roots:**")
                    for r in roots:
                        st.write(f"- $\\lambda = {format_complex(r)}$")
                        
                if edu_mode and not A.is_symbolic:
                    st.info("💡 Note: Finding roots of polynomials numerically can be highly unstable for large matrices. Small floating-point differences may artificially separate repeated roots.")
            except Exception as e:
                st.error(f"Polynomial Error: {e}")

    with tab3:
        st.subheader("Spectral Identities & Diagonalizability")
        if A and st.button("Run Spectral Diagnostics"):
            try:
                eigen_data = EigenEngine.compute_eigen_data(A, tol)
                
                # Trace / Determinant
                sum_v, prod_v, s_eig, p_eig, tr, det = EigenDiagnostics.verify_trace_det(A, eigen_data, tol)
                
                st.markdown("#### Trace Identity: $\\sum \\lambda_i = \\text{Trace}(A)$")
                st.write(f"Sum of Eigenvalues: `{format_complex(s_eig)}`")
                st.write(f"Trace of A: `{format_complex(tr)}`")
                if sum_v: st.success("✅ Trace Identity Holds.")
                else: st.error("❌ Trace Identity Failed.")
                
                st.markdown("#### Determinant Identity: $\\prod \\lambda_i = \\det(A)$")
                st.write(f"Product of Eigenvalues: `{format_complex(p_eig)}`")
                st.write(f"Determinant of A: `{format_complex(det)}`")
                if prod_v: st.success("✅ Determinant Identity Holds.")
                else: st.error("❌ Determinant Identity Failed.")
                
                st.markdown("---")
                st.markdown("#### Diagonalizability ($A = PDP^{-1}$)")
                is_diag, reason = EigenDiagnostics.check_diagonalizability(A, eigen_data)
                if is_diag: st.success(reason)
                else: st.error(reason)
                
            except Exception as e:
                st.error(f"Diagnostic Error: {e}")

    with tab4:
        st.subheader("🎓 Educational Learning Mode")
        with st.expander("1. What are Eigenvalues and Eigenvectors?"):
            st.write("For a square matrix $A$, a non-zero vector $v$ is an eigenvector if multiplying it by $A$ only scales its length, without changing its span. The scaling factor is the eigenvalue $\\lambda$. Mathematically: $Av = \\lambda v$.")
        with st.expander("2. Algebraic vs Geometric Multiplicity"):
            st.write("""
            - **Algebraic Multiplicity (AM):** How many times $\\lambda$ appears as a root of the characteristic polynomial.
            - **Geometric Multiplicity (GM):** The dimension of the Eigenspace $E_\\lambda = \\text{Null}(A - \\lambda I)$. (The number of independent eigenvectors for $\\lambda$).
            - **Theorem:** $1 \\leq GM \\leq AM$. If $GM < AM$, the matrix is *defective* and cannot be diagonalized.
            """)
        with st.expander("3. Why do some real matrices have complex eigenvalues?"):
            st.write("A real matrix might encode a rotation (e.g., a 2D rotation matrix). Because no real vector keeps its direction under a pure rotation, the eigenvalues and eigenvectors must live in the complex plane to satisfy $Av = \\lambda v$.")

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
