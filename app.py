import streamlit as st
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.matrix.validation import ValidationEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix.advanced_operations import AdvancedMatrixOperations
from linear_algebra.matrix.decompositions import MatrixDecompositions
from linear_algebra.matrix.analysis import MatrixAnalysis
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
    This interactive laboratory explores core concepts in Linear Algebra.
    
    **M1 & M2 Features:**
    - Standard & Advanced Operations (Kronecker, Hadamard, Exponentials).
    - Decompositions (LU, Cholesky).
    - Matrix Analysis (Norms, Condition Numbers, Structure Tests).
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
        "Hadamard Product (A ∘ B)", 
        "Element-wise Division (A ./ B)", 
        "Kronecker Product (A ⊗ B)", 
        "Matrix Power (A^n)", 
        "Matrix Exponential exp(A)",
        "Scalar Shift (A + λI)"
    ])
    
    if adv_op == "Matrix Power (A^n)":
        n_val = st.number_input("Exponent n", value=2, step=1)
    if adv_op == "Scalar Shift (A + λI)":
        lambda_val = st.number_input("Lambda (λ)", value=1.0)
        
    if st.button("Compute Advanced"):
        try:
            if adv_op == "Hadamard Product (A ∘ B)":
                res = AdvancedMatrixOperations.hadamard_product(A, B)
                st.latex("A \\circ B = " + res.to_latex())
            elif adv_op == "Element-wise Division (A ./ B)":
                res = AdvancedMatrixOperations.elementwise_division(A, B, tol)
                st.latex("A ./ B = " + res.to_latex())
            elif adv_op == "Kronecker Product (A ⊗ B)":
                res = AdvancedMatrixOperations.kronecker_product(A, B)
                st.latex("A \\otimes B = " + res.to_latex())
            elif adv_op == "Matrix Power (A^n)":
                res = AdvancedMatrixOperations.matrix_power(A, n_val)
                st.latex(f"A^{{{n_val}}} = " + res.to_latex())
            elif adv_op == "Matrix Exponential exp(A)":
                res = AdvancedMatrixOperations.matrix_exponential(A)
                st.latex("e^A = " + res.to_latex())
            elif adv_op == "Scalar Shift (A + λI)":
                res = AdvancedMatrixOperations.scalar_shift(A, lambda_val)
                st.latex(f"A + {lambda_val}I = " + res.to_latex())
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
                st.latex("P = " + P.to_latex())
                st.latex("L = " + L.to_latex())
                st.latex("U = " + U.to_latex())
                if edu_mode:
                    is_val, res, PLU = ValidationEngine.verify_lu(A, P, L, U, tol)
                    st.write(f"**Residual $\vert{}\vert{}PLU - A\vert{}\vert{}$:** {res:e}")
                    
            elif dec_op == "Cholesky (A = LL^T)":
                L = MatrixDecompositions.cholesky_decomposition(A)
                st.latex("L = " + L.to_latex())
                L_T = MatrixEngine(L.matrix.T, L.is_symbolic)
                st.latex("L^T = " + L_T.to_latex())
        except Exception as e:
            st.error(f"Error: {e}")

elif choice == "Matrix Analysis":
    st.title("Matrix Analysis & Conditioning")
    A = parse_input(st.text_area("Matrix A", "1,2; 2,4.0001"))
    if A:
        st.latex("A = " + A.to_latex())
        
        st.subheader("Linear Independence")
        ind_report = MatrixAnalysis.linear_independence(A)
        for k, v in ind_report.items():
            st.write(f"- **{k}:** {v}")

        st.subheader("Matrix Structure")
        structs = MatrixAnalysis.structure_tests(A, tol)
        cols = st.columns(4)
        for idx, (k, v) in enumerate(structs.items()):
            cols[idx % 4].metric(k, "Yes" if v else "No")

        st.subheader("Norms & Conditioning (Numerical)")
        if not A.is_symbolic:
            norms = MatrixAnalysis.norms(A)
            for k, v in norms.items():
                st.write(f"- **{k}:** `{v:.6f}`")
            st.write(f"- **Condition Number $\\kappa(A)$:** `{MatrixAnalysis.condition_number(A):.6e}`")
            if edu_mode:
                st.info("💡 High condition number implies the matrix is ill-conditioned (near-singular).")

elif choice == "Linear Systems":
    st.title("System of Linear Equations: $Ax = b$")
    col1, col2 = st.columns([2, 1])
    with col1:
        A = parse_input(st.text_area("Coefficient Matrix A", "2,1,-1; -3,-1,2; -2,1,2"))
    with col2:
        b = parse_input(st.text_area("Constants Vector b", "8; -11; -3"))
        
    if A and b and st.button("Solve"):
        try:
            res, report = MatrixOperations.solve_system(A, b)
            st.write(f"**Status:** {report['status']}")
            if res: st.latex("x = " + res.to_latex())
            
            # Milestone 2: Residual check
            if "Unique" in report['status'] and not A.is_symbolic:
                Ax = MatrixOperations.multiply(A, res)
                residual = np.linalg.norm(Ax.matrix - b.matrix)
                st.write(f"**Solution Residual $\vert{}\vert{}Ax - b\vert{}\vert{}$:** `{residual:e}`")
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
    B = parse_input(st.text_area("Matrix B (for two-matrix identities)", "1,2; 3,4", key="ValB"))
    
    if A and st.button("Validate Identities"):
        st.subheader("1. Inverse Identity: $AA^{-1} \\approx I$")
        is_val, res_val, I_calc = ValidationEngine.verify_inverse(A, tol)
        st.write("✅ VERIFIED" if is_val else "❌ FAILED", f"(Residual: {res_val:e})")
        
        if B and A.cols == B.rows:
            st.subheader("2. Transpose Multiplication: $(AB)^T = B^T A^T$")
            is_val_t, res_val_t, LHS, RHS = ValidationEngine.verify_transpose_multiplication(A, B, tol)
            st.write("✅ VERIFIED" if is_val_t else "❌ FAILED", f"(Residual: {res_val_t:e})")
