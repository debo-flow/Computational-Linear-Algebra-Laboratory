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

# ================= NEW M6 IMPORTS =================
from linear_algebra.vector_spaces.basis import BasisAnalyzer
from linear_algebra.vector_spaces.dimension import DimensionAnalyzer
from linear_algebra.vector_spaces.coordinates import CoordinateSystem
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
    "Linear Systems Lab",           
    "Vector Spaces Lab",            
    "Basis & Coordinates Lab",      # NEW M6 TAB
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
    **M1-M6 Comprehensive Features:**
    - Matrix & Vector Core Arithmetic.
    - Consistency Analysis ($Ax = b$).
    - Vector Spaces (Spans, Linear Independence, Subspaces).
    - **[M6]** Basis Extraction, Coordinate Transformations ($[v]_B$), and Rank-Nullity Analysis.
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

# ... (Advanced Operations, Decompositions, Matrix Analysis, Determinants, Rank remain same as before) ...
# I am condensing the unchanged M1-M4 blocks for brevity, but they stay completely intact in your actual file.
# You already have these tabs perfectly configured from M4 & M5. I am skipping their raw text copy-paste to focus on the M6 block.

# ======================= NEW M6 BASIS & COORDINATES LAB =======================
elif choice == "Basis & Coordinates Lab":
    st.title("Basis, Coordinates & Dimension Laboratory")
    st.markdown("Extract non-redundant bases, transform coordinate systems, and validate the Rank-Nullity Theorem.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Basis Extraction & Validation", 
        "Rank-Nullity & Dimensions", 
        "Coordinates & Transformations",
        "Visualization"
    ])
    
    with tab1:
        st.subheader("Redundant Vector Removal & Basis Extraction")
        v_str = st.text_input("Generating Set (separated by | )", "1,2,3 | 2,4,6 | 1,0,1")
        if st.button("Extract Basis"):
            try:
                vectors = parse_vector_list(v_str)
                basis, pivots = BasisAnalyzer.extract_basis(vectors)
                
                st.write(f"**Original Set Size:** {len(vectors)}")
                st.write(f"**Extracted Basis Size:** {len(basis)}")
                st.write(f"**Retained Indices (1-based):** {[p+1 for p in pivots]}")
                
                for i, v in enumerate(basis):
                    st.latex(f"b_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
                    
                is_val, msg = BasisAnalyzer.validate_ambient_basis(basis, vectors[0].dim, tol)
                if is_val:
                    st.success(f"✅ {msg}")
                else:
                    st.warning(f"⚠️ Extracted set is a subspace basis, not an ambient basis: {msg}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.subheader("Dimension Analysis & Rank-Nullity Theorem")
        st.write("Theorem: $\\text{Rank}(A) + \\text{Nullity}(A) = n$ (Number of columns)")
        A_str = st.text_area("Matrix A", "1,2,3; 4,5,6")
        A = parse_input(A_str)
        if A and st.button("Analyze Dimensions"):
            try:
                report = DimensionAnalyzer.analyze_matrix_spaces(A, tol)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"- **Domain ($n$):** $\\mathbb{{R}}^{{{report['ambient_domain_dim']}}}$")
                    st.write(f"- **Rank (Col Space Dim):** {report['rank']}")
                    st.write(f"- **Row Space Dim:** {report['row_space_dim']}")
                with col2:
                    st.write(f"- **Codomain ($m$):** $\\mathbb{{R}}^{{{report['ambient_codomain_dim']}}}$")
                    st.write(f"- **Nullity (Null Space Dim):** {report['nullity']}")
                    
                if report['rank_nullity_holds']:
                    st.success(f"✅ Rank-Nullity Holds: {report['rank']} + {report['nullity']} = {report['ambient_domain_dim']}")
                
                st.markdown("---")
                st.subheader("Row Space Basis")
                row_basis = BasisAnalyzer.get_row_space_basis(A)
                for i, v in enumerate(row_basis):
                    st.latex(f"r_{i+1} = [{', '.join([str(x) for x in v.vector])}]")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.subheader("Coordinate Representation & Change of Basis")
        st.write("Transform coordinates between standard space and custom bases.")
        
        mode = st.radio("Select Transformation", ["Find Coordinates $[v]_B$", "Change of Basis Matrix $P_{C \\leftarrow B}$"])
        
        if mode == "Find Coordinates $[v]_B$":
            b_str = st.text_input("Ordered Basis $B$ (separated by | )", "1,1 | 1,-1")
            tgt_str = st.text_input("Target Vector $v$ in standard coordinates", "3,1")
            
            if st.button("Compute Coordinates"):
                try:
                    basis = parse_vector_list(b_str)
                    target = parse_vector_list(tgt_str)[0]
                    coords, res = CoordinateSystem.get_coordinates(basis, target, tol)
                    
                    st.latex("[v]_B = " + coords.to_latex())
                    st.write(f"Reconstruction Residual $\vert{}\vert{}P[v]_B - v\vert{}\vert{}$: `{res:e}`")
                except Exception as e:
                    st.error(f"Coordinate Error: {e}")
                    
        else:
            b_str = st.text_input("Source Basis $B$ (separated by | )", "1,0 | 0,1")
            c_str = st.text_input("Target Basis $C$ (separated by | )", "1,1 | -1,1")
            
            if st.button("Compute Transformation Matrix"):
                try:
                    basis_B = parse_vector_list(b_str)
                    basis_C = parse_vector_list(c_str)
                    
                    P_CB = CoordinateSystem.change_of_basis_matrix(basis_B, basis_C, tol)
                    st.latex("P_{C \\leftarrow B} = " + P_CB.to_latex())
                    if edu_mode:
                        st.info("💡 To transform a vector, multiply: $[v]_C = P_{C \\leftarrow B} [v]_B$")
                except Exception as e:
                    st.error(f"Transformation Error: {e}")

    with tab4:
        st.subheader("Basis Visualization (3D)")
        st.write("Visualize the basis vectors to understand the space they span.")
        b3_str = st.text_input("3D Ordered Basis $B$ (separated by | )", "1,0,0 | 0,1,0 | 1,1,1")
        if st.button("Plot Basis"):
            try:
                basis = parse_vector_list(b3_str)
                if basis[0].dim != 3:
                    st.error("Visualization is only supported for 3D vectors.")
                else:
                    st.plotly_chart(plot_3d_vectors(basis, [f"b_{i+1}" for i in range(len(basis))]))
                    st.info("If fewer than 3 vectors are supplied, they span a plane or a line within 3D space.")
            except Exception as e:
                st.error(f"Plot Error: {e}")
