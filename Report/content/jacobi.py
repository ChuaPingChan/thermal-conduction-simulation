def jacobi(A, b, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = np.zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = np.diagflat(np.diag(A))
    D_inv = np.diagflat(1/np.diag(A))
    C = D - A

    # Iterate
    x_history = x[np.newaxis,:]
    while not stabalised(x_history, machine_epsilon):
        x = np.diag((b + np.dot(C,x)) * D_inv)
        x_history = np.concatenate( (x_history,x[np.newaxis,:]) )
    return x
