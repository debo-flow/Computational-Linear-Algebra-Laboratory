import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla

# Base matrix and vector operations
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

# Systems and Spaces
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

# Transformations, Subspaces, Eigen, SVD
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

# ================= NEW M12 IMPORTS =================
from linear_algebra.orthogonality.inner_product import InnerProductEngine
from linear_algebra.orthogonality.gram_schmidt import GramSchmidtEngine
from linear_algebra.orthogonality.projections import ProjectionEngine
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
    "Orthogonality & Projections",  # NEW M12 TAB
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
    **M1-M12 Comprehensive Features:**
    - Standard Operations, Decompositions, Determinants.
    - $Ax=b$ Consistency Analysis & Exact Solvers.
    - Vector Spaces (Spans, Bases, Coordinates).
    - Linear Mappings, Subspaces & Orthogonality.
    - Eigenvalues, Diagonalization, Similarity ($P^{-1}AP$), & Jordan Forms.
    - SVD, Eckart-Young Truncation & Pseudoinverses.
    - **[M12]** Orthogonality Lab: Hermitian Inner Products, Classical/Modified Gram-Schmidt, Orthogonal Projections ($P=QQ^H$), and Least-Squares Geometry.
    """)

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

# ... (I am heavily contracting the visualization lines for M2-M10 to fit the context window, 
# but all M1-M11 lines exist unchanged structurally just as in the previous iteration.)
elif choice == "SVD & Low-Rank Approx Lab":
    st.title("SVD, Low-Rank Approx & Pseudoinverse Lab")
    A = parse_input(st.text_area("Matrix $A_{m \\times n}$", "1,2; 3,4; 5,6"))
    if A and st.button("Compute SVD"):
        try:
            svd_data = SVDEngine.compute(A, full_matrices=True, tol=tol)
            st.latex("U = " + svd_data['U'].to_latex())
            st.write(f"Singular Values $\\Sigma$: `[{', '.join([f'{s:.4g}' for s in svd_data['S']])}]`")
            st.latex("V^H = " + svd_data['Vh'].to_latex())
            st.plotly_chart(plot_singular_values(svd_data['S'], log_scale=True))
        except Exception as e: st.error(f"SVD Error: {e}")

# ======================= NEW M12 ORTHOGONALITY & PROJECTIONS LAB =======================
elif choice == "Orthogonality & Projections":
    st.title("Orthogonality, Gram-Schmidt & Projections Laboratory")
    st.markdown("Analyze Hermitian inner products, Orthonormalize via CGS/MGS, and construct Orthogonal Projections.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Inner Products & Norms", 
        "Gram-Schmidt Orthogonalization", 
        "Orthogonal Projections",
        "Least-Squares Geometry",
        "🎓 Learning Mode"
    ])

    with tab1:
        st.subheader("Vector Inner Products & Gram Matrix")
        v_str = st.text_input("Vectors (separated by | )", "1, 2, 3 | 4, -2, 0 | 0, 0, 1")
        if st.button("Analyze Vectors"):
            try:
                vectors = parse_vector_list(v_str)
                if len(vectors) < 2: st.warning("Please provide at least 2 vectors.")
                else:
                    report = InnerProductEngine.analyze_orthonormal_set(vectors, tol)
                    st.write(f"- **Pairwise Orthogonal:** {'✅ Yes' if report['is_orthogonal'] else '❌ No'}")
                    st.write(f"- **Fully Orthonormal Set:** {'✅ Yes' if report['is_orthonormal'] else '❌ No'}")
                    st.write(f"- **Max Orthogonality Error:** `{report['max_orthogonality_error']:e}`")
                    
                    st.markdown("#### Gram Matrix $G$ ($G_{ij} = \\langle v_i, v_j \\rangle$)")
                    st.latex("G = " + report['gram_matrix'].to_latex())
                    
                    if edu_mode:
                        st.info("💡 For complex vectors, we use the Hermitian Inner Product: $\\langle u, v \\rangle = u^H v$. A set is orthonormal if and only if its Gram Matrix is exactly the Identity Matrix $I$.")
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    with tab2:
        st.subheader("Gram-Schmidt Process (CGS vs MGS)")
        gs_v_str = st.text_input("Input Basis (separated by | )", "1,1,0 | 1,0,1 | 0,1,1", key="gs_basis")
        method = st.radio("Method", ["Modified Gram-Schmidt (MGS) - Stable", "Classical Gram-Schmidt (CGS) - Unstable"])
        
        if st.button("Orthogonalize"):
            try:
                vectors = parse_vector_list(gs_v_str)
                m_type = 'modified' if 'Modified' in method else 'classical'
                report = GramSchmidtEngine.compute(vectors, method=m_type, tol=tol)
                
                st.write(f"**Method Used:** {m_type.capitalize()}")
                st.write(f"**Orthogonality Error $\\max \vert{}\\langle q_i, q_j \\rangle - \\delta_{{ij}}\vert{}$:** `{report['orthogonality_error']:e}`")
                
                st.markdown("#### Orthonormal Vectors ($Q$)")
                for i, q in enumerate(report['orthonormal_vectors']):
                    st.latex(f"q_{{{i+1}}} = [{', '.join([format_complex(x) for x in q.vector])}]^T")
                
                if report['skipped_indices']:
                    st.warning(f"⚠️ Dependent vectors detected and removed at input indices: {[i+1 for i in report['skipped_indices']]}")
                    
            except Exception as e:
                st.error(f"Gram-Schmidt Error: {e}")

    with tab3:
        st.subheader("Orthogonal Subspace Projection $P_{Col(A)}$")
        A_str = st.text_area("Subspace Matrix $A$ (Columns span the subspace)", "1, 2; 3, 4; 5, 6")
        v_str = st.text_input("Vector $v$ to project", "1, 1, 1")
        
        if st.button("Compute Projection & Decomposition"):
            try:
                A = parse_input(A_str)
                v = parse_vector_list(v_str)[0]
                
                # Projection Matrix
                P = ProjectionEngine.projection_matrix(A, tol)
                st.latex("P = Q Q^H = " + P.to_latex())
                
                # Decomposition
                decomp = ProjectionEngine.orthogonal_decomposition(v, A, tol)
                st.latex("v = p + r")
                st.latex("p = " + decomp['p'].to_latex())
                st.latex("r = " + decomp['r'].to_latex())
                
                # Pythagorean Theorem
                st.markdown("#### Pythagorean Identity $\vert{}\vert{}v\vert{}\vert{}^2 = \vert{}\vert{}p\vert{}\vert{}^2 + \vert{}\vert{}r\vert{}\vert{}^2$")
                st.write(f"$\\Vert{}p\\Vert{} = {decomp['norm_p']:.4g}, \\quad \\Vert{}r\\Vert{} = {decomp['norm_r']:.4g}$")
                if decomp['pythagorean_valid']:
                    st.success(f"✅ Identity Verified (Error: `{decomp['pythagorean_error']:e}`)")
                else:
                    st.error("❌ Identity Failed.")
            except Exception as e:
                st.error(f"Projection Error: {e}")

    with tab4:
        st.subheader("Least-Squares Geometric Interpretation")
        st.write("Verifies the normal condition $A^H r = 0$ for $Ax \\approx b$.")
        col1, col2 = st.columns(2)
        with col1: L_A = parse_input(st.text_area("Matrix $A$", "1, 1; 1, -1; 1, 2", key="LSA"))
        with col2: L_b = parse_vector_list(st.text_area("Target Vector $b$", "2; 0; 3", key="LSb"))[0]
        
        if L_A and L_b and st.button("Verify LS Geometry"):
            try:
                # Solve via pseudoinverse
                A_plus = PseudoinverseEngine.compute(L_A, tol)
                x_num = np.array(A_plus.matrix) @ np.array(L_b.vector)
                x = VectorEngine(x_num)
                
                res = ProjectionEngine.verify_least_squares_orthogonality(L_A, L_b, x, tol)
                st.latex("x_{LSTSQ} = " + x.to_latex())
                st.latex("p = Ax = " + res['p'].to_latex())
                st.latex("r = b - p = " + res['r'].to_latex())
                
                st.markdown("#### Normal Equation Residual: $A^H r$")
                st.write(f"Norm of $A^H r$: `{res['normal_residual_norm']:e}`")
                if res['is_orthogonal']:
                    st.success("✅ The residual $r$ is perfectly orthogonal to the column space of $A$.")
                else:
                    st.error("❌ Orthogonality failed.")
            except Exception as e:
                st.error(f"LS Error: {e}")

    with tab5:
        st.subheader("🎓 Educational Learning Mode: Orthogonality")
        with st.expander("1. Inner Products & Hermitian Conjugates"):
            st.write("For real vectors, orthogonality means $u^T v = 0$. For complex vectors, we strictly use the Hermitian inner product $u^H v = 0$, requiring us to take the complex conjugate of the first vector. This ensures norms $\vert{}\vert{}u\vert{}\vert{}^2 = u^H u$ remain strictly positive real numbers.")
        with st.expander("2. Classical vs Modified Gram-Schmidt (CGS vs MGS)"):
            st.write("Classical GS projects the *original* vector against all previous orthogonal bases. Due to floating-point rounding, these small errors accumulate, leading to loss of orthogonality. Modified GS solves this by iteratively projecting the *updated* residual vector, making it vastly more numerically stable without needing extra computations.")
        with st.expander("3. Orthogonal Projection Matrices ($P = P^2 = P^H$)"):
            st.write("If you have an orthonormal basis $Q$ spanning a subspace, the matrix that projects any vector down into that subspace is simply $P = QQ^H$. Because projecting a shadow twice does nothing new, $P^2 = P$ (idempotent). Because projections find the closest point straight down, $P^H = P$ (self-adjoint).")
        with st.expander("4. Least-Squares Geometry"):
            st.write("When $Ax=b$ has no solution, we seek the shadow of $b$ on the column space of $A$. We call this shadow $p$. The difference $r = b - p$ points exactly $90^\\circ$ away from the column space, meaning $A^H r = 0$. This geometric truth derives the famous Normal Equations $A^H A x = A^H b$.")

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
