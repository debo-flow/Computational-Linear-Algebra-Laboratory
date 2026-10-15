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

# ================= NEW M8 IMPORTS =================
from linear_algebra.subspaces.fundamental import FundamentalSubspaces
from linear_algebra.subspaces.orthogonality import SubspaceOrthogonality
from linear_algebra.subspaces.comparison import SubspaceComparison
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
    "Fundamental Subspaces Lab",    # NEW M8 TAB
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

if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1-M8 Comprehensive Features:**
    - Standard & Advanced Operations, LU/Cholesky Decompositions.
    - $Ax=b$ Consistency Analysis & Parametric Solvers.
    - Vector Spaces, Spans, Bases, Coordinates $[v]_B$.
    - Linear Mappings $T(x) = Ax$, Injectivity/Surjectivity.
    - **[M8]** The Four Fundamental Subspaces, Orthogonal Complements, and Rank-Nullity Verification.
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
            if operation == "Addition (A+B)":
                st.latex("A + B = " + MatrixOperations.add(A, B).to_latex())
            elif operation == "Multiplication (AB)":
                st.latex("A B = " + MatrixOperations.multiply(A, B).to_latex())
            elif operation == "Determinant det(A)":
                st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
            elif operation == "Inverse A⁻¹":
                st.latex("A^{-1} = " + MatrixOperations.inverse(A, tol).to_latex())
        except Exception as e:
            st.error(f"Mathematical Error: {e}")

elif choice == "Advanced Operations":
    st.title("Advanced Matrix Operations")
    col1, col2 = st.columns(2)
    with col1:
        A = parse_input(st.text_area("Matrix A", "2,1; 1,2", key="AdvA"))
        if A: st.latex("A = " + A.to_latex())
    with col2:
        B = parse_input(st.text_area("Matrix B (if needed)", "1,0; 0,1", key="AdvB"))
        if B: st.latex("B = " + B.to_latex())
        
    adv_op = st.selectbox("Operation", [
        "Hadamard Product (A ∘ B)", "Element-wise Division (A ./ B)", 
        "Kronecker Product (A ⊗ B)", "Matrix Power (A^n)", 
        "Matrix Exponential exp(A)", "Scalar Shift (A + λI)"
    ])
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
        if not A.is_square():
            st.error("These operations require a square matrix.")
        else:
            if st.button("Compute Full Analysis"):
                try:
                    det = MatrixOperations.determinant(A, tol)
                    st.latex(f"\\det(A) = {det}")
                    C = MinorsCofactors.cofactor_matrix(A)
                    st.latex("C = " + C.to_latex())
                    adj = MinorsCofactors.adjugate(A)
                    st.latex("\\text{adj}(A) = " + adj.to_latex())
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
            if 'singular_values' in report: st.write(f"- **Singular Values ($\\sigma$):** {np.round(report['singular_values'], 6)}")
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
            
            if report['solution_type'] != "none":
                if st.button("Compute Exact / Parametric Solution"):
                    sol = SystemSolvers.parametric_solve(system)
                    st.latex("X = " + sol.to_latex())
        except Exception as e:
            st.error(f"System Error: {e}")

elif choice == "Vector Spaces Lab":
    st.title("Vector Spaces, Span & Subspaces Laboratory")
    st.markdown("Explore combinations, linear independence, spans, and fundamental subspaces (Null / Column).")
    
    st.subheader("Span Membership Analyzer")
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
    st.title("Basis, Coordinates & Dimension Laboratory")
    
    st.subheader("Redundant Vector Removal & Basis Extraction")
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
    st.title("Linear Transformations & Representation Lab")
    
    t_type = st.selectbox("Transformation Type", ["2D Rotation", "2D Scaling", "2D Reflection", "2D Shear", "Custom Matrix"])
    A = None
    if t_type == "2D Rotation":
        theta = st.number_input("Angle $\\theta$ (Degrees)", value=45.0)
        A = GeometricTransformations.rotation_2d(theta, is_degrees=True)
    elif t_type == "Custom Matrix":
        c_str = st.text_area("Enter Matrix A", "1,2; 3,4", key="cusA")
        A = parse_input(c_str)

    if A and st.button("Plot 2D Transformation"):
        st.latex("T(x) = Ax = " + A.to_latex() + " x")
        if A.shape == (2, 2) and not A.is_symbolic:
            st.plotly_chart(plot_2d_transformation(A.matrix))

# ======================= NEW M8 FUNDAMENTAL SUBSPACES LAB =======================
elif choice == "Fundamental Subspaces Lab":
    st.title("The Four Fundamental Subspaces Laboratory")
    st.markdown("Analyze $Col(A)$, $Null(A)$, $Row(A)$, and $LeftNull(A)$ alongside Orthogonality and Rank-Nullity.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Subspace Extractor", 
        "Rank-Nullity Theorems", 
        "Orthogonal Complements",
        "Subspace Comparison",
        "🎓 Learning Mode"
    ])

    A_str = st.text_area("Target Matrix $A_{m \\times n}$", "1,2,3; 4,5,6; 7,8,9")
    A = parse_input(A_str)

    with tab1:
        if A and st.button("Analyze All Four Subspaces"):
            try:
                report = FundamentalSubspaces.analyze(A, tol)
                
                st.latex(f"A \\in \\mathbb{{R}}^{{{report['dimensions'][0]} \\times {report['dimensions'][1]}}}")
                st.write(f"**Rank:** {report['rank']}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("1. Column Space $Col(A)$")
                    st.write(f"**Dimension:** {report['col_space']['dim']} $\\quad$ (Pivots: {[p+1 for p in report['col_space']['pivots']]})")
                    for i, v in enumerate(report['col_space']['basis']):
                        st.latex(f"c_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                        
                    st.subheader("3. Row Space $Row(A)$")
                    st.write(f"**Dimension:** {report['row_space']['dim']}")
                    for i, v in enumerate(report['row_space']['basis']):
                        st.latex(f"r_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                with c2:
                    st.subheader("2. Null Space $Null(A)$")
                    st.write(f"**Dimension (Nullity):** {report['null_space']['dim']}")
                    for i, v in enumerate(report['null_space']['basis']):
                        st.latex(f"n_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                    if report['null_space']['dim'] > 0 and not A.is_symbolic:
                        st.write(f"Max Verification Residual $\vert{}\vert{}Av\vert{}\vert{}$: `{report['null_space']['max_residual']:e}`")

                    st.subheader("4. Left Null Space $Null(A^T)$")
                    st.write(f"**Dimension:** {report['left_null_space']['dim']}")
                    for i, v in enumerate(report['left_null_space']['basis']):
                        st.latex(f"l_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.subheader("Rank-Nullity Verification")
        if A and st.button("Verify Rank-Nullity"):
            report = FundamentalSubspaces.analyze(A, tol)
            m, n = report['dimensions']
            rank = report['rank']
            nullity = report['nullity']
            left_nullity = report['left_nullity']
            
            st.markdown("### Domain (Columns) Theorem")
            st.latex("\\text{Rank}(A) + \\text{Nullity}(A) = n")
            st.latex(f"{rank} + {nullity} = {n}")
            if report['rank_nullity_domain']: st.success("✅ Theorem holds for domain.")
            
            st.markdown("### Codomain (Rows) Theorem")
            st.latex("\\text{Rank}(A) + \\text{Nullity}(A^T) = m")
            st.latex(f"{rank} + {left_nullity} = {m}")
            if report['rank_nullity_codomain']: st.success("✅ Theorem holds for codomain.")

    with tab3:
        st.subheader("Orthogonal Complement Verification")
        st.write("Verifies $Row(A) \\perp Null(A)$ and $Col(A) \\perp LeftNull(A)$.")
        if A and st.button("Verify Orthogonality"):
            report = FundamentalSubspaces.analyze(A, tol)
            
            # Row(A) vs Null(A)
            st.markdown("### $Row(A) \\perp Null(A)$")
            if report['row_space']['dim'] == 0 or report['null_space']['dim'] == 0:
                st.info("Trivial orthogonality (one of the subspaces is the zero vector).")
            else:
                is_ortho, res = SubspaceOrthogonality.verify_orthogonality(report['row_space']['basis'], report['null_space']['basis'], tol)
                if is_ortho: st.success("✅ Verified Orthogonal.")
                else: st.error("❌ Orthogonality failed.")
                st.write(f"Max Inner Product Residual: `{res:e}`")

            # Col(A) vs LeftNull(A)
            st.markdown("### $Col(A) \\perp LeftNull(A)$")
            if report['col_space']['dim'] == 0 or report['left_null_space']['dim'] == 0:
                st.info("Trivial orthogonality (one of the subspaces is the zero vector).")
            else:
                is_ortho, res = SubspaceOrthogonality.verify_orthogonality(report['col_space']['basis'], report['left_null_space']['basis'], tol)
                if is_ortho: st.success("✅ Verified Orthogonal.")
                else: st.error("❌ Orthogonality failed.")
                st.write(f"Max Inner Product Residual: `{res:e}`")

    with tab4:
        st.subheader("Subspace Basis Comparator")
        st.write("Determine if two sets of basis vectors mathematically span the exact same subspace.")
        u_str = st.text_input("Basis U Vectors (separated by | )", "1,0,0 | 0,1,0")
        v_str = st.text_input("Basis V Vectors (separated by | )", "1,1,0 | 1,-1,0")
        
        if st.button("Compare Spans"):
            try:
                basis_U = parse_vector_list(u_str)
                basis_V = parse_vector_list(v_str)
                comp_report = SubspaceComparison.compare_bases(basis_U, basis_V, tol)
                
                st.write(f"- Dimension of Span(U): **{comp_report['dim_U']}**")
                st.write(f"- Dimension of Span(V): **{comp_report['dim_V']}**")
                st.write(f"- Dimension of Span(U ∪ V): **{comp_report['dim_sum_UV']}**")
                st.write(f"- Computed Intersection Dimension: **{comp_report['dim_intersection']}**")
                
                if comp_report['spans_same']:
                    st.success("✅ The two bases span the EXACT same subspace.")
                else:
                    st.error("❌ The bases span DIFFERENT subspaces.")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab5:
        st.subheader("🎓 Learning Mode: The Four Subspaces")
        with st.expander("1. What are the four fundamental subspaces?"):
            st.write("For any $m \\times n$ matrix $A$, there are four tightly linked spaces: Column Space, Null Space, Row Space, and Left Null Space. Together, they decompose the domain $\\mathbb{R}^n$ and codomain $\\mathbb{R}^m$ into orthogonal counterparts.")
        with st.expander("2. Orthogonal Complements"):
            st.write("The fundamental theorem of linear algebra states that the Row Space is the orthogonal complement to the Null Space in $\\mathbb{R}^n$, meaning every row vector is perpendicular to every vector in the null space ($Ax=0$). Similarly, the Column Space is the orthogonal complement to the Left Null Space in $\\mathbb{R}^m$.")
        with st.expander("3. Why are original columns used for Col(A)?"):
            st.write("Row operations preserve row dependence, but they destroy column dependence. Therefore, we use Row Reduction (RREF) strictly to find the *indices* of the pivot columns, and then we go back to the original matrix $A$ to pull those exact columns as our basis for $Col(A)$.")
# ==============================================================================

elif choice == "Vector Calculator":
    st.title("3D Vector Laboratory")
    v1_str = st.text_input("Vector u", "1, 2, 3")
    v2_str = st.text_input("Vector v", "4, 5, 6")
    try:
        u = VectorEngine([float(x) for x in v1_str.split(',')])
        v = VectorEngine([float(x) for x in v2_str.split(',')])
        st.latex(f"\\mathbf{{u}} \\cdot \\mathbf{{v}} = {VectorEngine.dot_product(u, v)}")
        st.plotly_chart(plot_3d_vectors([u, v, VectorEngine.cross_product(u, v)], ["u", "v", "u x v"]))
    except Exception as e:
        st.error(f"Error: {e}")

elif choice == "Validation Lab":
    st.title("Validation Laboratory")
    A = parse_input(st.text_area("Matrix A", "4,7; 2,6", key="ValA"))
    if A and st.button("Validate $AA^{-1} = I$"):
        is_val, res_val, I_calc = ValidationEngine.verify_inverse(A, tol)
        st.write("✅ VERIFIED" if is_val else "❌ FAILED", f"(Residual: {res_val:e})")
