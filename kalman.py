import numpy as np
import matplotlib.pyplot as plt


def simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng):
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    dim_x = A.shape[0]
    dim_y = H.shape[0]

    x0_mean = np.asarray(x0_mean).reshape(dim_x, 1)
    P0 = np.atleast_2d(P0)

    X = np.zeros((N + 1, dim_x, 1))
    Y = np.zeros((N, dim_y, 1))

    x0 = rng.multivariate_normal(mean=x0_mean.ravel(), cov=P0).reshape(dim_x, 1)
    X[0] = x0

    for n in range(1, N + 1):
        w = rng.multivariate_normal(mean=np.zeros(dim_x), cov=Q).reshape(dim_x, 1)
        X[n] = A @ X[n - 1] + w

    for n in range(1, N + 1):
        v = rng.multivariate_normal(mean=np.zeros(dim_y), cov=R).reshape(dim_y, 1)
        Y[n - 1] = H @ X[n] + v

    return X.squeeze(-1), Y.squeeze(-1)


def kalman_filter(A, H, Q, R, x0_hat, P0, Y):
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    Y = np.atleast_2d(Y)
    N = Y.shape[0]
    dim_x = A.shape[0]

    x_hat = np.asarray(x0_hat).reshape(dim_x, 1)
    P = np.atleast_2d(P0)

    x_hat_filt = np.zeros((N + 1, dim_x, 1))
    P_filt = np.zeros((N + 1, dim_x, dim_x))

    x_hat_filt[0] = x_hat
    P_filt[0] = P

    I = np.eye(dim_x)

    for n in range(1, N + 1):
        # Predict
        x_pred = A @ x_hat
        P_pred = A @ P @ A.T + Q

        # Update
        y = Y[n - 1].reshape(-1, 1)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innovation = y - (H @ x_pred)
        x_hat = x_pred + K @ innovation
        P = (I - K @ H) @ P_pred

        x_hat_filt[n] = x_hat
        P_filt[n] = P

    return x_hat_filt.squeeze(-1), P_filt


def main():
    rng = np.random.default_rng(7)

    # --- 2D financial model: price + drift ---
    # x_n = [p_n, mu_n]^T
    # p_n = p_{n-1} + mu_{n-1} + noise
    # mu_n = mu_{n-1} + noise
    A = np.array([
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    H = np.array([[1.0, 0.0]])  # observe price only

    # Process noise: price shock and drift shock
    sigma_p = 0.40   # "efficient price" volatility per step
    sigma_mu = 0.05  # drift changes slowly
    Q = np.array([
        [sigma_p**2, 0.0],
        [0.0, sigma_mu**2]
    ])

    # Observation noise: microstructure / measurement noise
    sigma_y = 0.80
    R = np.array([[sigma_y**2]])

    # Prior
    x0_mean = np.array([10.0, 10.05])
    P0 = np.array([
        [2.0, 0.0],
        [0.0, 0.2]
    ])

    N = 120

    # Simulate and filter
    X, Y = simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng)
    Xhat, _ = kalman_filter(A, H, Q, R, x0_hat=x0_mean, P0=P0, Y=Y)

    t = np.arange(N + 1)

    # --- Plot 1: observed price vs true latent price vs filtered latent price ---
    plt.figure()
    plt.plot(t, X[:, 0], label="True latent price p_n")
    plt.plot(t[1:], Y[:, 0], label="Observed price y_n")
    plt.plot(t, Xhat[:, 0], label="Filtered price estimate")
    plt.xlabel("n")
    plt.ylabel("price (arbitrary units)")
    plt.title("Kalman filter (2D state): price")
    plt.legend()
    plt.show()

    # --- Plot 2: drift (trend) true vs filtered estimate ---
    plt.figure()
    plt.plot(t, X[:, 1], label="True drift mu_n")
    plt.plot(t, Xhat[:, 1], label="Filtered drift estimate")
    plt.xlabel("n")
    plt.ylabel("drift")
    plt.title("Kalman filter (2D state): drift")
    plt.legend()
    plt.show()

    # --- Plot 3: 2D state-space trajectory (p vs mu) ---
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], label="True state trajectory (p, mu)")
    plt.plot(Xhat[:, 0], Xhat[:, 1], label="Filtered state trajectory (p, mu)")
    plt.xlabel("price p")
    plt.ylabel("drift mu")
    plt.title("2D hidden state trajectory (state space)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
