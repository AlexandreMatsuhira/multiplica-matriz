import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from servers.computation_server import ComputationServer

def verify_fix():
    print("Verifying fix for MemoryError (Sliced Matrix A)...")
    
    # Setup
    server = ComputationServer("test_server")
    
    # Create random matrices
    m, n, p = 100, 50, 20
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
    
    # Define a slice for the server
    start_row = 20
    end_row = 40
    rows = end_row - start_row
    
    # Slice A
    A_slice = A[start_row:end_row]
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    print(f"Slice: {start_row} to {end_row} ({rows} rows)")
    print(f"Sliced A shape: {A_slice.shape}")
    
    # Expected result (using local multiplication on the slice)
    expected_C_slice = np.dot(A_slice, B)
    
    # Actual result using ComputationServer logic
    # Note: We pass the SLICED A, but the GLOBAL start_row and end_row
    # The server should handle this correctly now.
    result_list = server.compute_partial_multiplication(
        A_slice.tolist(), 
        B.tolist(), 
        start_row, 
        end_row, 
        block_size=16
    )
    
    actual_C_slice = np.array(result_list)
    
    print(f"Result shape: {actual_C_slice.shape}")
    
    # Verification
    if actual_C_slice.shape != expected_C_slice.shape:
        print(f"❌ Shape mismatch! Expected {expected_C_slice.shape}, got {actual_C_slice.shape}")
        sys.exit(1)
        
    if np.allclose(actual_C_slice, expected_C_slice):
        print("✅ Verification PASSED: Results match!")
    else:
        print("❌ Verification FAILED: Results do not match!")
        diff = np.abs(actual_C_slice - expected_C_slice)
        print(f"Max difference: {np.max(diff)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_fix()
