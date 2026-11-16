import numpy as np
import scipy.fft as fft

class SpectralMethods:
    @staticmethod
    def fourier_derivative_1d(u, L):
        """
        Computes the spatial derivative of periodic u using Fourier Spectral Method.
        du/dx = IFFT( i * k * FFT(u) )
        """
        N = len(u)
        u_hat = fft.fft(u)
        
        # Wavenumbers
        k = (2 * np.pi / L) * np.fft.fftfreq(N, d=1.0/N)
        
        # Spectral derivative
        du_hat = 1j * k * u_hat
        du_dx = np.real(fft.ifft(du_hat))
        
        return du_dx
