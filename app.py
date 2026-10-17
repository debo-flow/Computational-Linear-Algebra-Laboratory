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

# ================= NEW M10 IMPORTS =================
from linear_algebra.diagonalization.core import DiagonalizationEngine
from linear_algebra.diagonalization.similarity import SimilarityAnalyzer
from linear_algebra.diagonalization.functions import MatrixFunctions
from linear_algebra.diagonalization.jordan import JordanCanonicalForm
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
    "Matrix Diagonalization Lab",   # NEW M10 TAB
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
    **M1-M10 Comprehensive Features:**
    - Standard Operations, Decompositions, Determinants.
    - $Ax=b$ Consistency Analysis & Exact Solvers.
    - Vector Spaces (Spans, Bases, Coordinates).
    - Linear Mappings, Subspaces & Orthogonality.
    - Eigenvalues, Characteristic Polynomials & AM/GM Multiplicities.
    - **[M10]** Matrix Diagonalization ($A=PDP^{-1}$), Similarity Transformations ($P^{-1}AP$), Matrix Exponentials, and exact Jordan Canonical Forms.
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
        B_str = st.text_area("Constants Vector B", "8; -11; -3")
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
            st.write(f"**Rank:** {report['rank']}, **Nullity:** {report['nullity']}, **Left Nullity:** {report['left_nullity']}")
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Eigenvalues & Polynomials":
    st.title("Eigenvalues, Eigenvectors & Characteristic Polynomial")
    A_str = st.text_area("Square Matrix $A$", "1, 2; 2, 1")
    A = parse_input(A_str)
    if A and st.button("Compute Eigen-decomposition"):
        try:
            eigen_data = EigenEngine.compute_eigen_data(A, tol)
            for i, data in enumerate(eigen_data):
                st.markdown(f"### $\\lambda_{i+1} = {format_complex(data['eigenvalue'])}$")
                st.write(f"- **AM:** {data['algebraic_multiplicity']}, **GM:** {data['geometric_multiplicity']}")
        except Exception as e:
            st.error(f"Calculation Error: {e}")

# ======================= NEW M10 MATRIX DIAGONALIZATION LAB =======================
elif choice == "Matrix Diagonalization Lab":
    st.title("Matrix Diagonalization & Jordan Form Laboratory")
    st.markdown("Analyze Diagonalizability ($A = PDP^{-1}$), Similarity ($B = P^{-1}AP$), and exact Jordan Forms.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Diagonalization $A=PDP^{-1}$", 
        "Similarity Transforms $B=P^{-1}AP$", 
        "Matrix Powers & Exponentials",
        "Jordan Canonical Form (Symbolic)",
        "🎓 Learning Mode"
    ])

    A_str = st.text_area("Square Matrix $A$", "4, -2; 1, 1")
    A = parse_input(A_str)

    with tab1:
        st.subheader("Matrix Diagonalizability Analysis")
        if A and st.button("Diagonalize Matrix"):
            try:
                report = DiagonalizationEngine.diagonalize(A, tol)
                if report['is_diagonalizable']:
                    st.success(report['message'])
                    st.latex("P = " + report['P'].to_latex())
                    st.latex("D = " + report['D'].to_latex())
                    st.latex("P^{-1} = " + report['P_inv'].to_latex())
                    
                    if not A.is_symbolic:
                        st.write(f"**Condition Number of P $\\kappa(P)$:** `{report['condition_number']:.2e}`")
                        if report['condition_number'] > 1e10:
                            st.warning("⚠️ P is ill-conditioned. The eigenvectors are nearly linearly dependent.")
                else:
                    st.error(report['message'])
                    if not A.is_symbolic and report['condition_number']:
                        st.write(f"**Condition Number of P $\\kappa(P)$:** `{report['condition_number']:.2e}`")
            except Exception as e:
                st.error(f"Diagonalization Error: {e}")

    with tab2:
        st.subheader("Similarity Transformation $B = P^{-1}AP$")
        P_str = st.text_area("Invertible Matrix $P$", "1, 1; 0, 1", key="simP")
        P = parse_input(P_str)
        if A and P and st.button("Apply Transformation"):
            try:
                B, P_inv = SimilarityAnalyzer.transform(A, P, tol)
                st.latex("B = P^{-1}AP = " + B.to_latex())
                
                # Verification
                tr_A = np.trace(A.matrix) if not A.is_symbolic else sp.trace(sp.Matrix(A.matrix))
                tr_B = np.trace(B.matrix) if not B.is_symbolic else sp.trace(sp.Matrix(B.matrix))
                det_A = MatrixOperations.determinant(A, tol)
                det_B = MatrixOperations.determinant(B, tol)
                
                st.write("**Invariant Properties Verification:**")
                st.write(f"- $\\text{{Tr}}(A)$: `{format_complex(tr_A)}` $\\quad \\text{{Tr}}(B)$: `{format_complex(tr_B)}`")
                st.write(f"- $\\det(A)$: `{format_complex(det_A)}` $\\quad \\det(B)$: `{format_complex(det_B)}`")
                
                if edu_mode:
                    st.info("💡 Similar matrices represent the same linear mapping under two different choices of basis. Therefore, their Trace, Determinant, and Eigenvalues are always strictly identical.")
            except Exception as e:
                st.error(f"Similarity Error: {e}")

    with tab3:
        st.subheader("Matrix Functions via Diagonalization")
        st.write("If $A = PDP^{-1}$, then $f(A) = P f(D) P^{-1}$.")
        
        func_choice = st.radio("Function", ["Matrix Power ($A^n$)", "Matrix Exponential ($e^A$)"])
        n_val = 2
        if func_choice == "Matrix Power ($A^n$)":
            n_val = st.number_input("Power $n$", value=5, min_value=0, step=1)
            
        if A and st.button("Compute Function"):
            try:
                report = DiagonalizationEngine.diagonalize(A, tol)
                if not report['is_diagonalizable']:
                    st.error("Matrix is strictly defective. Cannot compute function via Diagonalization.")
                else:
                    if func_choice == "Matrix Power ($A^n$)":
                        res = MatrixFunctions.matrix_power_diag(report['P'], report['D'], report['P_inv'], n_val)
                        st.latex(f"A^{{{n_val}}} = " + res.to_latex())
                    else:
                        res = MatrixFunctions.matrix_exp_diag(report['P'], report['D'], report['P_inv'])
                        st.latex("e^A = " + res.to_latex())
                        
                        # Compare to scipy.linalg.expm for numerics
                        if not A.is_symbolic:
                            scipy_res = spla.expm(A.matrix)
                            diff = np.linalg.norm(res.matrix - scipy_res)
                            st.write(f"Difference from direct `scipy.linalg.expm`: `{diff:.2e}`")
                            
            except Exception as e:
                st.error(f"Function Error: {e}")

    with tab4:
        st.subheader("Exact Jordan Canonical Form (JCF)")
        st.write("Defective matrices cannot be diagonalized into $D$. Instead, they are similar to a block-diagonal matrix $J$ containing Jordan Blocks.")
        if A and st.button("Compute Jordan Form"):
            if not A.is_symbolic:
                st.warning("⚠️ JCF is highly unstable numerically. Converting input to Exact Symbolic form for computation.")
                A = MatrixEngine(A.matrix.tolist(), True)
                
            try:
                P, J = JordanCanonicalForm.compute(A)
                st.latex("A = P J P^{-1}")
                st.latex("J = " + J.to_latex())
                st.latex("P = " + P.to_latex())
                
                if edu_mode:
                    st.info("💡 A Jordan block $J_k(\\lambda)$ places the eigenvalue $\\lambda$ on the diagonal, and $1$'s on the superdiagonal. A matrix is fully diagonalizable if and only if every Jordan block has size exactly $1 \\times 1$.")
            except Exception as e:
                st.error(f"Jordan Form Error: {e}")

    with tab5:
        st.subheader("🎓 Educational Learning Mode")
        with st.expander("1. What does Diagonalization tell us?"):
            st.write("Diagonalization separates the 'stretching' axes ($P$) from the stretch amounts ($D$). By rotating into the basis $P$, the matrix $A$ just acts like a simple scalar multiplier across its independent dimensions.")
        with st.expander("2. How does $P^{-1} A P$ relate to Coordinate Transformations?"):
            st.write("If $A$ represents a transformation in the standard basis, then $P^{-1} A P$ represents that exact same geometric transformation, but viewed from the perspective of the basis vectors contained in the columns of $P$.")
        with st.expander("3. Why is Jordan Canonical Form necessary?"):
            st.write("When a matrix has repeated eigenvalues and not enough eigenvectors (a defective matrix), it cannot be squashed all the way down to a diagonal line $D$. The JCF is the 'next best thing', placing $1$s right above the missing independent dimensions to account for the coupled shear behavior.")

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
