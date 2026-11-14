import numpy as np
import scipy.sparse as sps
import sys

class SparseMatrixEngine:
    @staticmethod
    def create_random(m, n, density=0.1, format='csr', seed=None):
        """Creates a random sparse matrix avoiding accidental densification."""
        if seed is not None:
            np.random.seed(seed)
        # Using scipy.sparse.random natively builds sparse coordinate data without allocating a dense matrix first
        mat = sps.random(m, n, density=density, format=format)
        return mat

    @staticmethod
    def convert_format(mat, target_format):
        """Converts sparse formats safely."""
        formats = {
            'coo': mat.tocoo,
            'csr': mat.tocsr,
            'csc': mat.tocsc,
            'dia': mat.todia,
            'lil': mat.tolil,
            'dok': mat.todok,
            'bsr': mat.tobsr
        }
        if target_format not in formats:
            raise ValueError(f"Unsupported target format: {target_format}")
        return formats[target_format]()

    @staticmethod
    def memory_analysis(mat):
        """Analyzes exact byte-storage complexity."""
        # Memory tracking of underlying arrays
        mem_bytes = 0
        arrays_info = {}
        
        if hasattr(mat, 'data') and mat.data is not None:
            mem_bytes += mat.data.nbytes
            arrays_info['data'] = mat.data.nbytes
        if hasattr(mat, 'indices') and mat.indices is not None:
            mem_bytes += mat.indices.nbytes
            arrays_info['indices'] = mat.indices.nbytes
        if hasattr(mat, 'indptr') and mat.indptr is not None:
            mem_bytes += mat.indptr.nbytes
            arrays_info['indptr'] = mat.indptr.nbytes
        if hasattr(mat, 'row') and mat.row is not None:
            mem_bytes += mat.row.nbytes
            arrays_info['row'] = mat.row.nbytes
        if hasattr(mat, 'col') and mat.col is not None:
            mem_bytes += mat.col.nbytes
            arrays_info['col'] = mat.col.nbytes
            
        # Theoretical dense memory (float64 = 8 bytes)
        m, n = mat.shape
        dense_bytes = m * n * 8
        
        nnz = mat.nnz
        density = nnz / (m * n)
        
        return {
            "format": mat.format,
            "shape": (m, n),
            "nnz": nnz,
            "density": density,
            "sparsity_ratio": 1.0 - density,
            "sparse_bytes": mem_bytes,
            "dense_bytes_equivalent": dense_bytes,
            "arrays_info": arrays_info,
            "compression_ratio": dense_bytes / max(mem_bytes, 1)
        }

    @staticmethod
    def matrix_multiply(A_sparse, B_sparse):
        """Performs sparse matrix-matrix multiplication (SPGEMM)."""
        # A @ B natively respects sparsity in scipy
        return A_sparse @ B_sparse
