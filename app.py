import streamlit as st
import numpy as np
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

# ================= M4 IMPORTS =================
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.elimination import EliminationEngine
from linear_algebra.systems.verification import SystemVerification

# ================= NEW M5 IMPORTS =================
from linear_algebra.vector_spaces.linear_combinations import LinearCombinations
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.independence import LinearIndependenceAnalyzer
from linear_algebra.vector_spaces.subspace_analysis import SubspaceAnalyzer
from linear_algebra.vector_spaces.axiom_checker import AxiomExplorer
# ==================================================

st.set_page_config(page_title="Comp. LinAlg Lab", layout="wide")

st.sidebar.title("Navigation")
pages = [
    "Home", 
    "Matrix Calculator", 
    "Advanced Operations",
    "Decompositions",
    "Matrix Analysis",
    "Determinants & Adjugate",      
    "Rank & Singularity",           
    "Determinant Properties",       
    "Linear Systems Lab",           
    "Vector Spaces Lab",            # NEW M5 TAB
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

if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1, M2, M3, M4 & M5 Features:**
    - Standard & Advanced Operations (Kronecker, Exponentials).
    - Decompositions (LU, Cholesky).
    - Matrix Analysis (Norms, Condition Numbers, Structure Tests).
    - Determinant Properties, Minors, Cofactors, Adjugate.
    - Tolerance-aware Rank & Singularity (SVD) Analysis.
    - Advanced Linear Systems Lab (Parametric Solvers, LSTSQ, Elimination Steppers).
    - **[M5]** Vector Spaces Lab (Spans, Linear Independence, Subspaces, Axioms).
    - Consistency Analysis & Vector Geometry.
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
                res = MatrixOperations.add(A, B)
                st.latex("A + B = " + res.to_latex())
            elif operation == "Multiplication (AB)":
                res = MatrixOperations.multiply(A, B)
                st.latex("A B = " + res.to_latex())
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
            if adv_op == "Hadamard Product (A ∘ B)":
                st.latex("A \\circ B = " + AdvancedMatrixOperations.hadamard_product(A, B).to_latex())
            elif adv_op == "Element-wise Division (A ./ B)":
                st.latex("A ./ B = " + AdvancedMatrixOperations.elementwise_division(A, B, tol).to_latex())
            elif adv_op == "Kronecker Product (A ⊗ B)":
                st.latex("A \\otimes B = " + AdvancedMatrixOperations.kronecker_product(A, B).to_latex())
            elif adv_op == "Matrix Power (A^n)":
                st.latex(f"A^{{{n_val}}} = " + AdvancedMatrixOperations.matrix_power(A, n_val).to_latex())
            elif adv_op == "Matrix Exponential exp(A)":
                st.latex("e^A = " + AdvancedMatrixOperations.matrix_exponential(A).to_latex())
            elif adv_op == "Scalar Shift (A + λI)":
                st.latex(f"A + {lambda_val}I = " + AdvancedMatrixOperations.scalar_shift(A, lambda_val).to_latex())
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

elif choice == "Matrix Analysis":
    st.title("Matrix Analysis & Conditioning")
    A = parse_input(st.text_area("Matrix A", "1,2; 2,4.0001"))
    if A:
        st.latex("A = " + A.to_latex())
        st.subheader("Linear Independence")
        ind_report = MatrixAnalysis.linear_independence(A)
        for k, v in ind_report.items(): st.write(f"- **{k}:** {v}")
        if not A.is_symbolic:
            st.subheader("Norms & Conditioning (Numerical)")
            for k, v in MatrixAnalysis.norms(A).items(): st.write(f"- **{k}:** `{v:.6f}`")
            st.write(f"- **Condition Number $\\kappa(A)$:** `{MatrixAnalysis.condition_number(A):.6e}`")

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
                    
                    st.subheader("Cofactor Matrix ($C$)")
                    C = MinorsCofactors.cofactor_matrix(A)
                    st.latex("C = " + C.to_latex())
                    
                    st.subheader("Adjugate Matrix ($\\text{adj}(A) = C^T$)")
                    adj = MinorsCofactors.adjugate(A)
                    st.latex("\\text{adj}(A) = " + adj.to_latex())
                    
                    if edu_mode:
                        st.info("💡 **Identity Verification:** $A \\cdot \\text{adj}(A) = \\det(A) I$")
                        A_adj = MatrixOperations.multiply(A, adj)
                        st.latex("A \\cdot \\text{adj}(A) = " + A_adj.to_latex())
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
            if report['is_square']:
                st.write(f"- **Determinant:** {report['determinant']}")
        with col2:
            if 'singular_values' in report:
                st.write(f"- **Singular Values ($\\sigma$):** {np.round(report['singular_values'], 6)}")
            if 'condition_number' in report:
                st.write(f"- **Condition Number $\\kappa(A)$:** {report['condition_number']:.6e}")
        
        if edu_mode:
            st.info("💡 **Ill-conditioned vs Singular**: A matrix is strictly singular if it has a rank deficit. It is ill-conditioned if its condition number (ratio of max to min singular value) is extremely high, causing numerical instability in computers despite being mathematically invertible.")

elif choice == "Determinant Properties":
    st.title("Determinant Properties Explorer")
    A = parse_input(st.text_area("Matrix A", "3,1; 2,4", key="propA"))
    B = parse_input(st.text_area("Matrix B", "1,5; 0,2", key="propB"))
    
    if A and B and st.button("Verify Properties"):
        try:
            st.subheader("1. Transpose Identity: $\\det(A^T) = \\det(A)$")
            is_val, res, LHS, RHS = DeterminantProperties.verify_transpose(A, tol)
            st.write("✅ VERIFIED" if is_val else "❌ FAILED")
            st.latex(f"\\det(A^T) = {RHS} \\quad \\det(A) = {LHS}")
            
            st.subheader("2. Multiplicative Identity: $\\det(AB) = \\det(A)\\det(B)$")
            is_val_m, res_m, LHS_m, RHS_m = DeterminantProperties.verify_multiplication(A, B, tol)
            st.write("✅ VERIFIED" if is_val_m else "❌ FAILED")
            st.latex(f"\\det(AB) = {LHS_m} \\quad \\det(A)\\det(B) = {RHS_m}")
            
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Linear Systems Lab":
    st.title("Advanced Linear Systems Laboratory: $AX = B$")
    st.markdown("Solve, classify, and visualize systems including multiple RHS, parametric solutions, and least-squares.")
    
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
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Solution Classification", 
                "Numerical Solvers", 
                "Gaussian Elimination Steps",
                "Verification & Residuals"
            ])
            
            with tab1:
                st.subheader("Rouché-Capelli Classification")
                report = SystemClassifier.classify(system, tol)
                st.write(f"- **Rank(A):** {report['rank_A']}")
                st.write(f"- **Rank([A|B]):** {report['rank_aug']}")
                st.write(f"- **Unknowns ($n$):** {report['unknowns']}")
                
                if report['solution_type'] == "none":
                    st.error(f"Status: {report['status']}")
                elif report['solution_type'] == "unique":
                    st.success(f"Status: {report['status']}")
                else:
                    st.warning(f"Status: {report['status']}")
                
                if report['solution_type'] != "none":
                    if st.button("Compute Exact / Parametric Solution"):
                        sol = SystemSolvers.parametric_solve(system)
                        st.latex("X = " + sol.to_latex())
                        
            with tab2:
                st.subheader("Numerical Solvers")
                solver_type = st.radio("Select Method", ["Direct Solve (Square)", "Least Squares (Overdetermined/Noisy)"])
                if st.button("Solve Numerically"):
                    if solver_type == "Direct Solve (Square)":
                        res = SystemSolvers.direct_solve(system)
                        st.latex("X = " + res.to_latex())
                    else:
                        res, res_vals, rank, s = SystemSolvers.least_squares(system)
                        st.latex("X_{LSTSQ} = " + res.to_latex())
                        if len(res_vals) > 0:
                            st.write(f"Least Squares Sum of Squared Residuals: `{res_vals[0]:.4e}`")
                        
            with tab3:
                st.subheader("Educational View: Forward Elimination")
                if st.button("Generate Steps"):
                    steps = EliminationEngine.gaussian_elimination_steps(system)
                    for step in steps:
                        st.markdown(f"**{step['description']}**")
                        st.latex(step['matrix'].to_latex())
                        st.markdown("---")
                        
            with tab4:
                st.subheader("Residual & Accuracy Analyzer")
                st.write("Calculates the error matrix $R = AX - B$ for a given solution.")
                X_str = st.text_area("Enter a Solution Matrix X to verify", "2; 3; -1")
                X_test = parse_input(X_str)
                if X_test and st.button("Verify Solution"):
                    R, col_norms, total_norm = SystemVerification.compute_residual(system, X_test)
                    st.latex("R = AX - B = " + R.to_latex())
                    st.write(f"**Total Frobenius Norm $\vert{}\vert{}R\vert{}\vert{}_F$:** `{total_norm:e}`")
                    for i, n in enumerate(col_norms):
                        st.write(f"- Column {i+1} Residual Norm: `{n:e}`")
                    
                    if total_norm < tol:
                        st.success("✅ The solution mathematically satisfies the system within the given tolerance.")
                    else:
                        st.error("❌ The solution does NOT satisfy the system. Significant residuals detected.")
                        
        except Exception as e:
            st.error(f"System Error: {e}")

# ======================= NEW M5 VECTOR SPACES LAB =======================
elif choice == "Vector Spaces Lab":
    st.title("Vector Spaces, Span & Subspaces Laboratory")
    st.markdown("Explore combinations, linear independence, spans, and fundamental subspaces (Null / Column).")
    
    def parse_vector_list(text):
        try:
            vec_strs = text.split('|')
            return [VectorEngine([float(x) for x in v.split(',')]) for v in vec_strs if v.strip()]
        except Exception:
            return None

    tab1, tab2, tab3, tab4 = st.tabs([
        "Combinations & Axioms", 
        "Span & Target Vector", 
        "Linear Independence", 
        "Subspaces: Null & Column"
    ])
    
    with tab1:
        st.subheader("Linear Combinations ($c_1v_1 + \\dots + c_kv_k$)")
        c_str = st.text_input("Scalars (comma separated)", "2, -1")
        v_str = st.text_input("Vectors (separated by | )", "1,0,0 | 0,1,0")
        
        if st.button("Compute Combination"):
            try:
                scalars = [float(s) for s in c_str.split(',')]
                vectors = parse_vector_list(v_str)
                res = LinearCombinations.compute(scalars, vectors)
                st.write(f"**Resultant Vector:** `[{', '.join([str(x) for x in res.vector])}]^T`")
                
                st.markdown("---")
                st.subheader("Axiom Exploratory Checks")
                st.info("💡 Computations below show empirical checks of axioms for the space $\\mathbb{R}^n$, not universal formal proofs.")
                if len(vectors) >= 2:
                    u = vectors[0]
                    v = vectors[1]
                    w = vectors[2] if len(vectors) >= 3 else VectorEngine(np.zeros(u.dim).tolist())
                    c, d = scalars[0], scalars[1] if len(scalars) > 1 else 1.0
                    
                    checks = AxiomExplorer.explore_axioms(u, v, w, c, d)
                    for k, val in checks.items():
                        st.write(f"- {k}: {'✅ Holds' if val else '❌ Failed'}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.subheader("Span Membership Analyzer")
        span_v_str = st.text_input("Spanning Vectors (separated by | )", "1,2,3 | 4,5,6")
        target_str = st.text_input("Target Vector", "7,8,9")
        
        if st.button("Analyze Span Membership"):
            try:
                vectors = parse_vector_list(span_v_str)
                target = parse_vector_list(target_str)[0]
                in_span, report = SpanAnalyzer.is_in_span(target, vectors, tol)
                
                if in_span:
                    st.success("✅ The target vector is **INSIDE** the span.")
                    if edu_mode:
                        st.write("This means there exists a linear combination of the spanning vectors that produces the target vector. The system $Ax=b$ is consistent.")
                else:
                    st.error("❌ The target vector is **OUTSIDE** the span.")
                    if edu_mode:
                        st.write("The system $Ax=b$ is mathematically inconsistent. Rank([A|b]) > Rank(A).")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.subheader("Linear Independence Test")
        ind_v_str = st.text_input("Vectors to Test (separated by | )", "1,0,0 | 0,1,0 | 1,1,0")
        
        if st.button("Check Independence"):
            try:
                vectors = parse_vector_list(ind_v_str)
                report = LinearIndependenceAnalyzer.analyze(vectors, tol)
                
                if report['is_independent']:
                    st.success("✅ The vectors are **Linearly Independent**.")
                else:
                    st.error("❌ The vectors are **Linearly Dependent**.")
                
                st.write(f"**Explanation:** {report['message']}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab4:
        st.subheader("Fundamental Subspaces")
        st.write("Extract the basis vectors for the Null Space (Kernel) and Column Space of Matrix A.")
        A_str = st.text_area("Matrix A (for Subspace extraction)", "1,2; 3,6", key="SubA")
        A = parse_input(A_str)
        
        if A and st.button("Extract Subspaces"):
            try:
                col_basis = SubspaceAnalyzer.get_column_space(A)
                null_basis = SubspaceAnalyzer.get_null_space(A)
                
                st.write("**Column Space (Image) Basis:**")
                if not col_basis: st.write("Trivial (Zero Vector only)")
                for i, v in enumerate(col_basis):
                    st.latex(f"v_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                    
                st.write("**Null Space (Kernel) Basis:**")
                if not null_basis: st.write("Trivial (Zero Vector only)")
                for i, v in enumerate(null_basis):
                    st.latex(f"v_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                    
                if edu_mode:
                    st.info("💡 The Null Space represents the solution space to the homogeneous equation $Ax=0$. The Column Space defines all reachable vectors $b$ for which $Ax=b$ is consistent.")
            except Exception as e:
                st.error(f"Error extracting subspaces: {e}")
# ========================================================================

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
