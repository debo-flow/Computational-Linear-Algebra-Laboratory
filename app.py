import streamlit as st
import numpy as np
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.matrix.validation import ValidationEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from visualization.vector_plots import plot_3d_vectors
from visualization.matrix_plots import plot_matrix_heatmap

st.set_page_config(page_title="Comp. LinAlg Lab", layout="wide")

st.sidebar.title("Navigation")
pages = [
    "Home", "Matrix Calculator", "Matrix Properties", 
    "Linear Systems", "Vector Calculator", "Validation Lab"
]
choice = st.sidebar.radio("Go to", pages)

st.sidebar.markdown("---")
edu_mode = st.sidebar.checkbox("🎓 Enable Educational Mode", value=True)
precision = st.sidebar.slider("Numerical Tolerance (10^-x)", 1, 15, 9)
tol = 10**(-precision)

def parse_input(mat_str):
    """Safely parse user matrix input."""
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
    This interactive laboratory allows you to explore, compute, visualize, and validate core concepts in Linear Algebra.
    
    **Features:**
    - Rigorous Matrix & Vector mathematical engines.
    - Consistency Analysis for Systems of Equations.
    - Geometric visualizations.
    - Built-in mathematical validation.
    
    *Use the sidebar to navigate to a specific lab.*
    """)

elif choice == "Matrix Calculator":
    st.title("Matrix Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Matrix A")
        matA_str = st.text_area("Enter rows separated by ';', elements by ','", "1,2; 3,4", key="A")
        A = parse_input(matA_str)
        if A:
            st.latex("A = " + A.to_latex())
    
    with col2:
        st.subheader("Matrix B")
        matB_str = st.text_area("Enter rows separated by ';', elements by ','", "5,6; 7,8", key="B")
        B = parse_input(matB_str)
        if B:
            st.latex("B = " + B.to_latex())

    operation = st.selectbox("Operation", ["Addition (A+B)", "Multiplication (AB)", "Determinant det(A)", "Inverse A⁻¹"])
    
    if st.button("Compute"):
        try:
            if operation == "Addition (A+B)":
                if edu_mode:
                    st.info("💡 **Educational Mode:** Matrix addition is defined only for matrices of the exact same dimensions. Elements are added pairwise: $C_{ij} = A_{ij} + B_{ij}$.")
                res = MatrixOperations.add(A, B)
                st.latex("A + B = " + res.to_latex())
                
            elif operation == "Multiplication (AB)":
                if edu_mode:
                    st.info("💡 **Educational Mode:** For AB, the number of columns in A must equal rows in B. The element $C_{ij}$ is the dot product of row $i$ from A and column $j$ from B.")
                res = MatrixOperations.multiply(A, B)
                st.latex("A B = " + res.to_latex())
                
            elif operation == "Determinant det(A)":
                det = MatrixOperations.determinant(A, tol)
                st.latex(f"\\det(A) = {det}")
                if edu_mode and A.is_square():
                    st.info(f"💡 If $\\det(A) \\neq 0$, the matrix forms a basis for $\\mathbb{{R}}^{A.rows}$ and is invertible.")
                    
            elif operation == "Inverse A⁻¹":
                res = MatrixOperations.inverse(A, tol)
                st.latex("A^{-1} = " + res.to_latex())
                
        except Exception as e:
            st.error(f"Mathematical Error: {e}")

elif choice == "Linear Systems":
    st.title("System of Linear Equations: $Ax = b$")
    st.markdown("Analyze and solve systems using rank consistency theorems.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        A_str = st.text_area("Coefficient Matrix A", "2,1,-1; -3,-1,2; -2,1,2")
        A = parse_input(A_str)
    with col2:
        b_str = st.text_area("Constants Vector b (Column)", "8; -11; -3")
        b = parse_input(b_str)
        
    if A and b:
        st.latex(f"A = {A.to_latex()}, \\quad b = {b.to_latex()}")
        if st.button("Solve System"):
            try:
                res, report = MatrixOperations.solve_system(A, b)
                
                st.subheader("Consistency Analysis")
                st.write(f"- $\\text{{rank}}(A) = {report['rank_A']}$")
                st.write(f"- $\\text{{rank}}([A|b]) = {report['rank_Aug']}$")
                st.write(f"- Unknowns ($n$) = {report['unknowns']}")
                
                if "Unique" in report['status']:
                    st.success(report['status'])
                    st.latex("x = " + res.to_latex())
                elif "No Solution" in report['status']:
                    st.error(report['status'])
                    if edu_mode:
                        st.info("Mathematically, $b$ is not in the column space of $A$.")
                else:
                    st.warning(report['status'])
                    st.latex("\\text{RREF}([A|b]) = " + res.to_latex())
                    
            except Exception as e:
                st.error(str(e))

elif choice == "Validation Lab":
    st.title("Validation Laboratory")
    st.markdown("Test computational results against formal mathematical identities.")
    
    A_str = st.text_area("Enter Matrix A for Validation", "4,7; 2,6")
    A = parse_input(A_str)
    
    if A:
        st.latex("A = " + A.to_latex())
        if st.button("Validate $A A^{-1} = I$"):
            is_valid, residual, I_calc = ValidationEngine.verify_inverse(A, tol)
            
            if is_valid:
                st.success("✅ NUMERICALLY VERIFIED")
            else:
                st.error("❌ NOT VERIFIED")
                
            st.write(f"**Numerical Residual (Error):** {residual:e}")
            st.latex("AA^{-1} \\approx " + I_calc.to_latex())

elif choice == "Vector Calculator":
    st.title("3D Vector Laboratory")
    v1_str = st.text_input("Vector u (e.g. 1,0,0)", "1, 2, 3")
    v2_str = st.text_input("Vector v (e.g. 0,1,0)", "4, 5, 6")
    
    try:
        u = VectorEngine([float(x) for x in v1_str.split(',')])
        v = VectorEngine([float(x) for x in v2_str.split(',')])
        
        st.latex(f"\\mathbf{{u}} \\cdot \\mathbf{{v}} = {VectorEngine.dot_product(u, v)}")
        cross = VectorEngine.cross_product(u, v)
        st.latex(f"\\mathbf{{u}} \\times \\mathbf{{v}} = [{', '.join([str(x) for x in cross.vector])}]^T")
        
        st.plotly_chart(plot_3d_vectors([u, v, cross], ["u", "v", "u x v"]))
    except Exception as e:
        st.error(f"Error parsing vectors: {e}")
