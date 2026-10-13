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

st.set_page_config(page_title="Comp. LinAlg Lab", layout="wide")

st.sidebar.title("Navigation")
pages = [
    "Home", 
    "Matrix Calculator", 
    "Advanced Operations",
    "Decompositions",
    "Matrix Analysis",
    "Determinants & Adjugate",      # NEW M3
    "Rank & Singularity",           # NEW M3
    "Determinant Properties",       # NEW M3
    "Linear Systems", 
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
    **M1, M2 & M3 Features:**
    - Standard & Advanced Operations (Kronecker, Exponentials).
    - Decompositions (LU, Cholesky).
    - Matrix Analysis (Norms, Condition Numbers, Structure Tests).
    - **[M3]** Determinant Properties, Minors, Cofactors, Adjugate.
    - **[M3]** Tolerance-aware Rank & Singularity (SVD) Analysis.
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

# ======================= MILESTONE 3 NEW TABS =======================

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

# ======================= EXISTING TABS CONTINUED =======================

elif choice == "Linear Systems":
    st.title("System of Linear Equations: $Ax = b$")
    col1, col2 = st.columns([2, 1])
    with col1: A = parse_input(st.text_area("Coefficient Matrix A", "2,1,-1; -3,-1,2; -2,1,2"))
    with col2: b = parse_input(st.text_area("Constants Vector b", "8; -11; -3"))
    if A and b and st.button("Solve"):
        try:
            res, report = MatrixOperations.solve_system(A, b)
            st.write(f"**Status:** {report['status']}")
            if res: st.latex("x = " + res.to_latex())
        except Exception as e:
            st.error(str(e))

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
