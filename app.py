import streamlit as st
import numpy as np
import sympy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import math
import time
import pandas as pd

# ================= CORE ENGINES (M1 - M25) =================
from linear_algebra.matrix.matrix_engine import MatrixEngine
from linear_algebra.matrix.operations import MatrixOperations
from linear_algebra.matrix.validation import ValidationEngine
from linear_algebra.vectors.vector_engine import VectorEngine
from linear_algebra.matrix.advanced_operations import AdvancedMatrixOperations
from linear_algebra.matrix.decompositions import MatrixDecompositions
from linear_algebra.matrix.analysis import MatrixAnalysis
from linear_algebra.matrix.minors_cofactors import MinorsCofactors
from linear_algebra.matrix.singularity_analysis import SingularityAnalyzer
from linear_algebra.systems.system_model import LinearSystem
from linear_algebra.systems.classification import SystemClassifier
from linear_algebra.systems.solvers import SystemSolvers
from linear_algebra.systems.verification import SystemVerification
from linear_algebra.vector_spaces.span_analysis import SpanAnalyzer
from linear_algebra.vector_spaces.basis import BasisAnalyzer
from linear_algebra.subspaces.fundamental import FundamentalSubspaces
from linear_algebra.eigenvalues.core import EigenEngine
from linear_algebra.diagonalization.core import DiagonalizationEngine
from linear_algebra.svd.core import SVDEngine
from linear_algebra.svd.approximation import LowRankApproximation
from linear_algebra.svd.pseudoinverse import PseudoinverseEngine
from linear_algebra.qr.core import QREngine
from linear_algebra.qr.least_squares import QRLeastSquares
from linear_algebra.factorizations.lu import LUEngine
from linear_algebra.factorizations.cholesky import CholeskyEngine
from linear_algebra.factorizations.triangular_solver import TriangularSolver
from linear_algebra.definiteness.classifier import DefinitenessClassifier
from linear_algebra.definiteness.sylvester import SylvesterCriterion
from linear_algebra.definiteness.quadratic_forms import QuadraticForms
from linear_algebra.definiteness.loewner import LoewnerOrder
from linear_algebra.definiteness.generator import DefiniteMatrixGenerator
from linear_algebra.numerical_eigen.iterative import IterativeEigenEngine
from linear_algebra.numerical_eigen.qr_algorithm import QREigenAlgorithm
from linear_algebra.numerical_eigen.schur import SchurDecomposition
from linear_algebra.numerical_eigen.sensitivity import EigenSensitivity
from linear_algebra.generalized_eigen.core import GeneralizedEigenEngine
from linear_algebra.generalized_eigen.pencil import MatrixPencilAnalyzer
from linear_algebra.generalized_eigen.rayleigh import GeneralizedRayleighQuotient
from linear_algebra.matrix_functions.core import MatrixFunctionsEngine
from linear_algebra.matrix_functions.diagnostics import FunctionDiagnostics
from linear_algebra.matrix_functions.ode_solver import LinearODESolver
from linear_algebra.calculus.autodiff import Dual, AutoDiffEngine
from linear_algebra.calculus.numerical import FiniteDifferenceEngine
from linear_algebra.calculus.symbolic import SymbolicCalculusEngine
from linear_algebra.calculus.matrix_rules import MatrixCalculusIdentities
from linear_algebra.calculus.optimization import OptimizationEngine
from linear_algebra.calculus.computational_graph.reverse_ad import Value
from linear_algebra.calculus.jvp_vjp import JVP_VJP_Engine
from linear_algebra.calculus.backend_ad import BackendAD
from linear_algebra.iterative_solvers.generators import MatrixGenerators
from linear_algebra.iterative_solvers.krylov import KrylovSubspace
from linear_algebra.iterative_solvers.solvers import IterativeSolvers
from linear_algebra.iterative_solvers.stationary import StationarySolvers
from linear_algebra.iterative_solvers.preconditioners import Preconditioners
from linear_algebra.optimization.functions import BenchmarkFunctions
from linear_algebra.optimization.unconstrained import UnconstrainedOptimization
from linear_algebra.optimization.constrained import ConstrainedOptimization
from linear_algebra.optimization.least_squares import NonlinearLeastSquares
from linear_algebra.integration_odes.integration import NumericalIntegration
from linear_algebra.integration_odes.ode_ivp import ODESolverIVP
from linear_algebra.integration_odes.ode_bvp import ODESolverBVP
from linear_algebra.integration_odes.pde_demo import PDEDemonstration
from linear_algebra.pde.mesh import Mesh1D, Mesh2D
from linear_algebra.pde.fdm import FiniteDifferencePDE
from linear_algebra.pde.fem import FiniteElementPDE
from linear_algebra.pde.fvm import FiniteVolumePDE
from linear_algebra.pde.spectral import SpectralMethods

# ================= VISUALIZATION ENGINES =================
from visualization.vector_plots import plot_3d_vectors
from visualization.matrix_plots import plot_matrix_heatmap
from visualization.transformation_plots import plot_2d_transformation
from visualization.svd_plots import plot_singular_values, plot_energy_retention
from visualization.quadratic_plots import plot_2d_quadratic_form
from visualization.eigen_plots import plot_eigenvalue_spectrum, plot_convergence_history
from visualization.generalized_eigen_plots import plot_generalized_eigenvalues
from visualization.matrix_functions_plots import plot_ode_time_series, plot_ode_phase_space
from visualization.calculus_plots import plot_optimization_path
from visualization.graph_plots import plot_computational_graph
from visualization.iterative_plots import plot_residual_history
from visualization.optimization_plots import plot_2d_optimization_trajectory
from visualization.integration_ode_plots import plot_numerical_integration, plot_ode_trajectories, plot_pde_heatmap
from visualization.pde_plots import plot_pde_surface_2d, plot_pde_evolution_1d

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
    "Orthogonality & Projections",
    "QR & Least-Squares Lab",
    "Advanced Factorizations Lab",
    "Pos-Def & Quadratic Forms",
    "Numerical Eigen & Schur Lab",
    "Singular & Gen Eigen Lab",
    "Matrix Functions Lab",
    "Matrix Calculus Lab",
    "Advanced AutoDiff & Graph Lab",
    "Iterative & Krylov Solvers Lab",
    "Optimization & Constraints Lab",
    "Integration & ODEs Lab",
    "Advanced PDEs & Physics Lab",
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

def get_f_eval(expr_str, var_name='x'):
    x_sym = sp.Symbol(var_name)
    expr = SymbolicCalculusEngine.safe_parse(expr_str)
    return sp.lambdify(x_sym, expr, "numpy"), expr

# ======================= HOME =======================
if choice == "Home":
    st.title("Welcome to the Computational Linear Algebra Laboratory 🧮")
    st.markdown("""
    **M1-M25 Comprehensive Features:**
    - Standard Operations, LU/Cholesky/QR/SVD Factorizations.
    - Consistency Analysis, Vector Spaces, Orthogonality ($P=QQ^H$).
    - Linear Mappings, Fundamental Subspaces, Pos-Def Geometries ($x^H A x$).
    - Iterative Eigenvalues, Diagonalization ($PDP^{-1}$), & Jordan Forms.
    - Sparse Iterative Solvers (CG, GMRES), Krylov Subspaces, Preconditioners.
    - Unconstrained/Constrained Optimization, Non-Linear Least Squares.
    - Numerical Integration, Adaptive ODEs (RK45, Stiff BDF), Boundary Value Problems.
    - **[M25]** Advanced PDE Lab: FDM, FEM, FVM, and Spectral Methods mapping continuously into Physics (Electrostatics, Wave propagation, Heat Diffusion).
    """)

# ======================= MATRIX CALCULATOR (M1-M2) =======================
elif choice == "Matrix Calculator":
    st.title("Standard Matrix Calculator")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "1,2; 3,4"))
    with col2: B = parse_input(st.text_area("Matrix B", "5,6; 7,8"))
    op = st.selectbox("Operation", ["Addition (A+B)", "Multiplication (AB)", "Determinant det(A)", "Inverse A⁻¹"])
    if st.button("Compute"):
        try:
            if op == "Addition (A+B)": st.latex("A+B = " + MatrixOperations.add(A, B).to_latex())
            elif op == "Multiplication (AB)": st.latex("AB = " + MatrixOperations.multiply(A, B).to_latex())
            elif op == "Determinant det(A)": st.latex(f"\\det(A) = {MatrixOperations.determinant(A, tol)}")
            elif op == "Inverse A⁻¹": st.latex("A^{-1} = " + MatrixOperations.inverse(A, tol).to_latex())
        except Exception as e: st.error(f"Error: {e}")

# ======================= ADVANCED OPERATIONS =======================
elif choice == "Advanced Operations":
    st.title("Advanced Matrix Operations")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "2,1; 1,2"))
    with col2: B = parse_input(st.text_area("Matrix B (if needed)", "1,0; 0,1"))
    adv_op = st.selectbox("Operation", ["Hadamard Product (A ∘ B)", "Element-wise Division (A ./ B)", "Kronecker Product (A ⊗ B)", "Matrix Power (A^n)"])
    n_val = st.number_input("Exponent n", value=2, step=1) if adv_op == "Matrix Power (A^n)" else None
    if st.button("Compute Advanced"):
        try:
            if adv_op == "Hadamard Product (A ∘ B)": st.latex(AdvancedMatrixOperations.hadamard_product(A, B).to_latex())
            elif adv_op == "Element-wise Division (A ./ B)": st.latex(AdvancedMatrixOperations.elementwise_division(A, B, tol).to_latex())
            elif adv_op == "Kronecker Product (A ⊗ B)": st.latex(AdvancedMatrixOperations.kronecker_product(A, B).to_latex())
            elif adv_op == "Matrix Power (A^n)": st.latex(AdvancedMatrixOperations.matrix_power(A, n_val).to_latex())
        except Exception as e: st.error(f"Error: {e}")

# ======================= DECOMPOSITIONS & DETERMINANTS (M3-M4) =======================
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
                st.latex("L = " + MatrixDecompositions.cholesky_decomposition(A).to_latex())
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

# ======================= LINEAR SYSTEMS & VECTOR SPACES (M4-M6) =======================
elif choice == "Linear Systems Lab":
    st.title("Linear Systems Laboratory: $AX = B$")
    col1, col2 = st.columns(2)
    with col1: A = parse_input(st.text_area("Matrix A", "2,1,-1; -3,-1,2; -2,1,2"))
    with col2: B = parse_input(st.text_area("Vector B", "8; -11; -3"))
    if A and B and st.button("Solve"):
        try:
            st.latex("X = " + SystemSolvers.parametric_solve(LinearSystem(A, B)).to_latex())
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
            st.success("✅ Inside span") if in_span else st.error("❌ Outside span")
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Basis & Coordinates Lab":
    st.title("Basis Extraction")
    v_str = st.text_input("Generating Set", "1,2,3 | 2,4,6 | 1,0,1")
    if st.button("Extract Basis"):
        try:
            basis, _ = BasisAnalyzer.extract_basis(parse_vector_list(v_str))
            for i, v in enumerate(basis): st.latex(f"b_{i+1} = [{', '.join([str(x) for x in v.vector])}]^T")
        except Exception as e: st.error(f"Error: {e}")

# ======================= TRANSFORMATIONS & SUBSPACES (M7-M8) =======================
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

# ======================= EIGENVALUES & DIAGONALIZATION (M9-M10) =======================
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

# ======================= SVD, ORTHOGONALITY & QR (M11-M13) =======================
elif choice == "SVD & Low-Rank Approx Lab":
    st.title("SVD, Low-Rank Approx & Pseudoinverse Lab")
    A = parse_input(st.text_area("Matrix $A_{m \\times n}$", "1,2; 3,4; 5,6"))
    if A and st.button("Compute SVD"):
        try:
            svd_data = SVDEngine.compute(A, full_matrices=True, tol=tol)
            st.latex("U = " + svd_data['U'].to_latex())
            st.write(f"Singular Values $\\Sigma$: `[{', '.join([f'{s:.4g}' for s in svd_data['S']])}]`")
            st.plotly_chart(plot_singular_values(svd_data['S'], log_scale=True))
        except Exception as e: st.error(f"Error: {e}")

elif choice == "Orthogonality & Projections":
    st.title("Orthogonality & Projections Laboratory")
    st.write("Analyzes Gram-Schmidt and Projection Matrices $P = QQ^H$.")
    gs_v_str = st.text_input("Input Basis", "1,1,0 | 1,0,1 | 0,1,1")
    if st.button("Orthogonalize"):
        report = GramSchmidtEngine.compute(parse_vector_list(gs_v_str), method='modified', tol=tol)
        for i, q in enumerate(report['orthonormal_vectors']):
            st.latex(f"q_{{{i+1}}} = [{', '.join([format_complex(x) for x in q.vector])}]^T")

elif choice == "QR & Least-Squares Lab":
    st.title("QR Decomposition & Least-Squares Laboratory")
    A = parse_input(st.text_area("Matrix A", "1,-1,4; 1,4,-2; 1,4,2; 1,-1,0"))
    if A and st.button("Compute QR"):
        qr_data = QREngine.compute_library_qr(A, mode='reduced', tol=tol)
        st.latex("Q = " + qr_data['Q'].to_latex())
        st.latex("R = " + qr_data['R'].to_latex())

# ======================= FACTORIZATIONS & QUADRATIC FORMS (M14-M15) =======================
elif choice == "Advanced Factorizations Lab":
    st.title("LU, Cholesky & Advanced Factorizations Laboratory")
    A = parse_input(st.text_area("Square Matrix", "4, 12, -16; 12, 37, -43; -16, -43, 98"))
    if A and st.button("Compute LU"):
        report = LUEngine.compute_partial_pivot(A, tol)
        st.latex("P = " + report['P'].to_latex())
        st.latex("L = " + report['L'].to_latex())

elif choice == "Pos-Def & Quadratic Forms":
    st.title("Definiteness, Quadratic Forms & Matrix Inequalities")
    A = parse_input(st.text_area("Hermitian Matrix A", "2, -1; -1, 2"))
    if A and st.button("Classify Matrix"):
        report = DefinitenessClassifier.classify_via_eigenvalues(A, tol)
        st.write(f"**Classification:** `{report['status']}`")

# ======================= EIGEN LABS (M16-M17) =======================
elif choice == "Numerical Eigen & Schur Lab":
    st.title("Numerical Eigenvalue Algorithms & Schur Decomposition")
    A = parse_input(st.text_area("Square Matrix $A$", "4, -1, 1; -1, 3, -2; 1, -2, 3"))
    if A and st.button("Compute Schur Decomposition"):
        schur_res = SchurDecomposition.compute(A, output='complex')
        st.latex("T = " + schur_res['T'].to_latex())
        st.latex("Z = " + schur_res['Z'].to_latex())

elif choice == "Singular & Gen Eigen Lab":
    st.title("Singular & Generalized Eigenvalue Problems ($Av = \\lambda Bv$)")
    A = parse_input(st.text_area("Matrix $A$", "2, -1; -1, 2"))
    B = parse_input(st.text_area("Matrix $B$", "1, 0; 0, 0"))
    if A and B and st.button("Compute General Eigenpairs"):
        results = GeneralizedEigenEngine.compute_general(A, B, tol)
        st.plotly_chart(plot_generalized_eigenvalues(results))

# ======================= MATRIX FUNCTIONS & CALCULUS (M18-M19) =======================
elif choice == "Matrix Functions Lab":
    st.title("Matrix Exponential, Logarithm & Powers Laboratory")
    A = parse_input(st.text_area("Square Matrix $A$", "0, -1; 1, 0"))
    if A and st.button("Compute $e^A$"):
        st.latex("e^A = " + MatrixFunctionsEngine.expm(A).to_latex())

elif choice == "Matrix Calculus Lab":
    st.title("Advanced Matrix Calculus & Differentiation Laboratory")
    expr_in = st.text_input("Scalar Function $f(x, y, z)$", "x**2 * y + sin(z)")
    if st.button("Compute Symbolic Gradient"):
        expr, grad_sym = SymbolicCalculusEngine.compute_gradient(expr_in, sp.symbols('x y z'))
        st.latex(f"\\nabla f = {sp.latex(grad_sym)}")

# ======================= AUTODIFF & GRAPHS (M20) =======================
elif choice == "Advanced AutoDiff & Graph Lab":
    st.title("Automatic Differentiation & Computational Graph Lab")
    val_x = st.number_input("Input x", value=2.0)
    val_y = st.number_input("Input y", value=3.0)
    if st.button("Build Graph & Backpropagate"):
        x = Value(val_x, label='x')
        y = Value(val_y, label='y')
        out = (x**2) * y + x.sin()
        out.backward()
        st.success(f"Forward Pass: {out.data:.5g}")
        st.write(f"Gradients: dx={x.grad:.5g}, dy={y.grad:.5g}")

# ======================= ITERATIVE SOLVERS (M22) =======================
elif choice == "Iterative & Krylov Solvers Lab":
    st.title("Advanced Iterative Solvers & Krylov Methods")
    st.write("Generates sparse matrices for CG, GMRES, BiCGSTAB.")
    if st.button("Generate 1D Poisson System"):
        A_sp = MatrixGenerators.poisson_1d(100)
        st.session_state['iter_A'] = A_sp
        st.session_state['iter_b'] = np.ones(100)
        st.success(f"Generated 100x100 sparse matrix.")

# ======================= OPTIMIZATION (M23) =======================
elif choice == "Optimization & Constraints Lab":
    st.title("Advanced Numerical Optimization Laboratory")
    f_name = st.selectbox("Benchmark Function", ["Rosenbrock", "Himmelblau", "Sphere"])
    x0_str = st.text_input("Initial Point $x_0$", "-1.2, 1.0")
    if st.button("Run BFGS Optimization"):
        x0 = np.array([float(v) for v in x0_str.split(',')])
        f_target = {"Rosenbrock": BenchmarkFunctions.rosenbrock, "Himmelblau": BenchmarkFunctions.himmelblau, "Sphere": BenchmarkFunctions.sphere}[f_name]
        x_opt, hist, conv = UnconstrainedOptimization.bfgs(f_target, x0, max_iter=500)
        st.write(f"Optimal Point Found: `({x_opt[0]:.4f}, {x_opt[1]:.4f})`")

# ======================= INTEGRATION & ODES (M24) =======================
elif choice == "Integration & ODEs Lab":
    st.title("Advanced Numerical Integration & Differential Equations")
    f_str = st.text_input("Integrand $f(x)$", "sin(x) + exp(-x)")
    a_val, b_val = st.columns(2)
    with a_val: a = st.number_input("a", value=0.0)
    with b_val: b = st.number_input("b", value=5.0)
    if st.button("Calculate Integral (Simpson's 1/3)"):
        f_func, f_expr = get_f_eval(f_str)
        val, nodes, y_vals = NumericalIntegration.simpsons_rule(f_func, a, b, 50)
        st.plotly_chart(plot_numerical_integration(f_func, a, b, nodes, y_vals, "Simpson's Rule"), use_container_width=True)
        st.success(f"Calculated Integral: `{val:.10f}`")

# ======================= PDES & PHYSICS (M25) =======================
elif choice == "Advanced PDEs & Physics Lab":
    st.title("Partial Differential Equations & Computational Physics")
    st.markdown("Simulate Continuous Geometries across FDM, FEM, FVM, and Spectral Discretizations.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Elliptic (2D Poisson FDM)", 
        "Parabolic (1D Heat FDM/FEM)", 
        "Hyperbolic (1D Wave Physics)",
        "Finite Volume (FVM) & Spectral",
        "🎓 Mathematical Stability (CFL)"
    ])

    with tab1:
        st.subheader("2D Poisson Equation (Electrostatics / Steady-State Heat)")
        c1, c2 = st.columns(2)
        with c1:
            nx = st.number_input("Grid X Resolution (Nx)", value=30, max_value=200)
            ny = st.number_input("Grid Y Resolution (Ny)", value=30, max_value=200)
        with c2:
            f_expr = st.text_input("Source Term $f(x,y)$", "sin(pi*x)*sin(pi*y)")
            
        if st.button("Solve 2D Poisson"):
            try:
                mesh = Mesh2D(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
                x_sym, y_sym = sp.symbols('x y')
                expr = SymbolicCalculusEngine.safe_parse(f_expr)
                f_func = sp.lambdify((x_sym, y_sym), expr, "numpy")
                def bc_func(x, y): return 0.0
                
                u_sol = FiniteDifferencePDE.solve_poisson_2d(mesh, f_func, bc_func)
                st.plotly_chart(plot_pde_surface_2d(mesh, u_sol, "2D Electrostatic Potential / Heat Surface"), use_container_width=True)
            except Exception as e:
                st.error(f"Poisson 2D Error: {e}")

    with tab2:
        st.subheader("1D Heat/Diffusion Equation")
        method = st.radio("Method", ["Finite Difference (Crank-Nicolson)", "Finite Element Method (Steady State)"])
        nx_1d = st.number_input("Nodes (Nx)", value=50, max_value=500)
                
        if st.button("Solve Diffusion System"):
            try:
                mesh = Mesh1D(L=1.0, nx=nx_1d)
                if "Crank" in method:
                    def u0(x): return np.exp(-100 * (x - 0.5)**2)
                    u_sol = FiniteDifferencePDE.solve_heat_1d_crank_nicolson(mesh, 1.0, 100, 0.01, u0, 0.0, 0.0)
                    st.plotly_chart(plot_pde_evolution_1d(mesh, np.linspace(0, 1.0, 101), u_sol, "Heat Diffusion"), use_container_width=True)
                else:
                    x_sym = sp.Symbol('x')
                    expr = SymbolicCalculusEngine.safe_parse("100 * sin(pi*x)")
                    f_func = sp.lambdify(x_sym, expr, "numpy")
                    u_sol = FiniteElementPDE.solve_poisson_1d(mesh, f_func)
                    st.write("1D FEM Solution Computed.")
            except Exception as e:
                st.error(f"Diffusion Error: {e}")

    with tab3:
        st.subheader("1D Wave Physics ($u_{tt} = c^2 u_{xx}$)")
        if st.button("Simulate Wave Propagation"):
            try:
                mesh = Mesh1D(L=1.0, nx=100)
                def u0(x): return np.exp(-200 * (x - 0.5)**2)
                def v0(x): return np.zeros_like(x)
                
                u_sol = FiniteDifferencePDE.solve_wave_1d(mesh, 1.0, 200, 1.0, u0, v0)
                st.plotly_chart(plot_pde_evolution_1d(mesh, np.linspace(0, 1.0, 201), u_sol, "Wave Propagation"), use_container_width=True)
            except Exception as e:
                st.error(f"Wave PDE Error: {e}")

    with tab4:
        st.subheader("Advanced Nonlinear & Conservation Methods")
        opt = st.radio("Simulation Focus", ["Nonlinear Burgers' Equation (FDM)", "Linear Advection (Finite Volume)"])
        if st.button("Run Simulation"):
            try:
                if "Burgers" in opt:
                    mesh = Mesh1D(2.0, 100)
                    def u0(x): return np.sin(np.pi * x)
                    u_sol = FiniteDifferencePDE.solve_burgers_1d(mesh, 0.5, 200, nu=0.01, u0_func=u0)
                    st.plotly_chart(plot_pde_evolution_1d(mesh, np.linspace(0, 0.5, 201), u_sol, "Burgers' Eq"), use_container_width=True)
                else:
                    mesh = Mesh1D(1.0, 100)
                    def u0(x): return np.exp(-100 * (x - 0.2)**2)
                    u_sol = FiniteVolumePDE.solve_advection_1d(mesh, 0.5, 200, velocity=1.0, u0_func=u0)
                    st.plotly_chart(plot_pde_evolution_1d(mesh, np.linspace(0, 0.5, 201), u_sol, "FVM Advection"), use_container_width=True)
            except Exception as e:
                st.error(f"Simulation Error: {e}")

    with tab5:
        st.subheader("🎓 Educational Theory: PDE Classification & Stability")
        with st.expander("The CFL Condition (Courant–Friedrichs–Lewy)"):
            st.write("In explicit hyperbolic solvers, numerical information can only travel one grid cell ($\\Delta x$) per time step ($\\Delta t$). Therefore, the physical wave speed $c$ must satisfy $c \\Delta t \\le \\Delta x$, meaning $\\text{CFL} \\le 1.0$.")

# ======================= UTILITIES =======================
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
    A = parse_input(st.text_area("Matrix A", "4,7; 2,6"))
    if A and st.button("Validate $AA^{-1} = I$"):
        is_val, res_val, I_calc = ValidationEngine.verify_inverse(A, tol)
        st.write("✅ VERIFIED" if is_val else "❌ FAILED", f"(Residual: {res_val:e})")
