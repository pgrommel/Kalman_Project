import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# Simulation + Kalman Filter
# -----------------------------
def simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng):
    """
    Simulate a linear Gaussian state-space model (LGSSM):

        X0 ~ N(x0_mean, P0)
        Xn = A X_{n-1} + Wn,  Wn ~ N(0, Q)
        Yn = H Xn + Vn,       Vn ~ N(0, R)

    Returns:
        X: shape (N+1, dim_x)
        Y: shape (N, dim_y)
    """
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    dim_x = A.shape[0]
    dim_y = H.shape[0]

    x0_mean = np.asarray(x0_mean).reshape(dim_x)
    P0 = np.atleast_2d(P0)

    X = np.zeros((N + 1, dim_x))
    Y = np.zeros((N, dim_y))

    X[0] = rng.multivariate_normal(mean=x0_mean, cov=P0)

    for n in range(1, N + 1):
        w = rng.multivariate_normal(mean=np.zeros(dim_x), cov=Q)
        X[n] = A @ X[n - 1] + w

    for n in range(1, N + 1):
        v = rng.multivariate_normal(mean=np.zeros(dim_y), cov=R)
        Y[n - 1] = H @ X[n] + v

    return X, Y


def kalman_filter(A, H, Q, R, x0_hat, P0, Y):
    """
    Standard Kalman filter for LGSSM.

    Returns:
        Xhat: shape (N+1, dim_x)   filtered estimates x_{n|n} (with Xhat[0] = x0_hat)
        P_filt: shape (N+1, dim_x, dim_x)   filtered covariances P_{n|n}
    """
    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    Y = np.atleast_2d(Y)
    N = Y.shape[0]
    dim_x = A.shape[0]

    x_hat = np.asarray(x0_hat).reshape(dim_x, 1)
    P = np.atleast_2d(P0)

    Xhat = np.zeros((N + 1, dim_x))
    P_filt = np.zeros((N + 1, dim_x, dim_x))

    Xhat[0] = x_hat.ravel()
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

        Xhat[n] = x_hat.ravel()
        P_filt[n] = P

    return Xhat, P_filt


# -----------------------------
# Plot helpers
# -----------------------------
def confidence_ellipse_points_2d(mean_2d, cov_2x2, conf=0.95, num_points=200):
    """
    Create points of a confidence ellipse for a 2D Gaussian.

    For df=2: chi2 quantiles (common):
        0.90 -> 4.605
        0.95 -> 5.991
        0.99 -> 9.210

    Returns:
        xs, ys: arrays with ellipse perimeter points
    """
    mean_2d = np.asarray(mean_2d).reshape(2)
    cov_2x2 = np.atleast_2d(cov_2x2)

    if cov_2x2.shape != (2, 2):
        raise ValueError("cov_2x2 must be shape (2, 2).")

    chi2_q = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}.get(conf, 5.991)
    scale = np.sqrt(chi2_q)

    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    eigvals = np.maximum(eigvals, 0.0)  # numeric safety
    transform = eigvecs @ np.diag(np.sqrt(eigvals) * scale)

    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle = np.vstack([np.cos(angles), np.sin(angles)])  # (2, num_points)
    ellipse = (transform @ circle).T + mean_2d  # (num_points, 2)

    return ellipse[:, 0], ellipse[:, 1]


def add_start_end_markers(fig, x, y, name_prefix, row, col):
    """
    Add start/end point markers for a trajectory.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    fig.add_trace(
        go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode="markers",
            name=f"{name_prefix} start",
            marker=dict(size=10, symbol="circle"),
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[y[-1]],
            mode="markers",
            name=f"{name_prefix} end",
            marker=dict(size=12, symbol="diamond"),
        ),
        row=row,
        col=col,
    )

def empirical_mse_mc(A, H, Q, R, x0_mean, P0, N, M, seed=0):
    """
    Monte Carlo estimate of MSE_n = E[ ||X_n - Xhat_{n|n}||^2 ], n=1..N
    using M independent simulated trajectories.

    Returns
    -------
    mse : (N+1,) array with mse[0] included (usually 0-ish / depends on init),
          mse[n] corresponds to time n.
    """
    rng = np.random.default_rng(seed)

    mse_acc = np.zeros(N + 1, dtype=float)

    for i in range(M):
        # Each run uses fresh randomness from rng (still reproducible overall via seed)
        X, Y = simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng)
        Xhat, _ = kalman_filter(A, H, Q, R, x0_hat=x0_mean, P0=P0, Y=Y)

        # squared Euclidean error at each time n (including n=0)
        err = X - Xhat                    # (N+1, dim_x)
        se = np.sum(err**2, axis=1)       # (N+1,)
        mse_acc += se

    mse = mse_acc / M
    return mse

# -----------------------------
# Plots (Dashboard + Separate)
# -----------------------------
def make_dashboard_plot(
    X,
    Y,
    Xhat,
    P_filt,
    out_html_path="kalman_2d_dashboard.html",
    out_png_prefix=None,
    ellipse_conf=0.95,
):
    """
    Creates a 3-panel Plotly dashboard and saves it as HTML.
    Adds start/end markers to trajectories and a confidence ellipse at the filtered end point.
    """
    N = Y.shape[0]
    t = np.arange(N + 1)
    t_obs = np.arange(1, N + 1)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            "Price: latent vs observed vs filtered",
            "Drift (trend): true vs filtered",
            "State space: (price, drift) trajectory",
        ),
    )

    # Row 1: price
    fig.add_trace(go.Scatter(x=t, y=X[:, 0], mode="lines", name="True latent price p_n"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_obs, y=Y[:, 0], mode="lines", name="Observed price y_n"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=Xhat[:, 0], mode="lines", name="Filtered price estimate"), row=1, col=1)
    fig.update_xaxes(title_text="n", row=1, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)

    # Row 2: drift
    fig.add_trace(go.Scatter(x=t, y=X[:, 1], mode="lines", name="True drift μ_n"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=Xhat[:, 1], mode="lines", name="Filtered drift estimate"), row=2, col=1)
    fig.update_xaxes(title_text="n", row=2, col=1)
    fig.update_yaxes(title_text="drift μ", row=2, col=1)

    # Row 3: state-space trajectories
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="lines+markers",
            name="True trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=Xhat[:, 0],
            y=Xhat[:, 1],
            mode="lines+markers",
            name="Filtered trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2),
        ),
        row=3,
        col=1,
    )

    # Start/end markers
    add_start_end_markers(fig, X[:, 0], X[:, 1], name_prefix="True", row=3, col=1)
    add_start_end_markers(fig, Xhat[:, 0], Xhat[:, 1], name_prefix="Filtered", row=3, col=1)

    # Confidence ellipse at filtered end point using P_{N|N}
    end_mean = Xhat[-1, :2]
    end_cov = P_filt[-1, :2, :2]
    ex, ey = confidence_ellipse_points_2d(end_mean, end_cov, conf=ellipse_conf, num_points=250)

    fig.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            name=f"{int(ellipse_conf * 100)}% confidence ellipse (end)",
            line=dict(width=2, dash="dash"),
        ),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="price p", row=3, col=1)
    fig.update_yaxes(title_text="drift μ", row=3, col=1)

    fig.update_layout(
        title="Kalman Filter (2D State Financial Model): Price + Drift",
        template="plotly_white",
        height=1000,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=60),
    )

    fig.write_html(out_html_path, include_plotlyjs="cdn")
    print(f"Wrote interactive dashboard to: {out_html_path}")

    if out_png_prefix is not None:
        try:
            fig.write_image(f"{out_png_prefix}_dashboard.png", scale=2)
            print(f"Wrote PNG to: {out_png_prefix}_dashboard.png")
        except Exception as e:
            print("PNG export failed (likely missing 'kaleido').")
            print('Install it via: pip install -U kaleido')
            print(f"Error was: {e}")

    return fig


def save_separate_plots(X, Y, Xhat, P_filt, prefix="kalman_2d", ellipse_conf=0.95):
    """
    Save 3 separate PNG figures (requires kaleido):
      1) Price time series
      2) Drift time series
      3) State-space trajectory (with start/end markers + end confidence ellipse)
    """
    N = Y.shape[0]
    t = np.arange(N + 1)
    t_obs = np.arange(1, N + 1)

    # 1) Price
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=t, y=X[:, 0], mode="lines", name="True latent price p_n"))
    fig_price.add_trace(go.Scatter(x=t_obs, y=Y[:, 0], mode="lines", name="Observed price y_n"))
    fig_price.add_trace(go.Scatter(x=t, y=Xhat[:, 0], mode="lines", name="Filtered estimate p̂_{n|n}"))
    fig_price.update_layout(
        title=dict(text="Latent Price Estimation via Kalman Filter", x=0.5),
        xaxis_title="Time index n",
        yaxis_title="Price level",
        template="plotly_white",
        height=500,
        width=900,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=80, b=120),
    )
    fig_price.write_image(f"{prefix}_price.png", scale=3)

    # 2) Drift
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Scatter(x=t, y=X[:, 1], mode="lines", name="True drift μ_n"))
    fig_drift.add_trace(go.Scatter(x=t, y=Xhat[:, 1], mode="lines", name="Filtered drift μ̂_{n|n}"))
    fig_drift.update_layout(
        title=dict(text="Trend (Drift) Estimation via Kalman Filter", x=0.5),
        xaxis_title="Time index n",
        yaxis_title="Drift μ",
        template="plotly_white",
        height=500,
        width=900,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=80, b=120),
    )
    fig_drift.write_image(f"{prefix}_drift.png", scale=3)

    # 3) State space
    fig_state = go.Figure()
    fig_state.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="lines+markers",
            name="True trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2),
        )
    )
    fig_state.add_trace(
        go.Scatter(
            x=Xhat[:, 0],
            y=Xhat[:, 1],
            mode="lines+markers",
            name="Filtered trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2),
        )
    )

    # Start/end markers
    fig_state.add_trace(go.Scatter(x=[X[0, 0]], y=[X[0, 1]], mode="markers", name="True start",
                                   marker=dict(size=10, symbol="circle")))
    fig_state.add_trace(go.Scatter(x=[X[-1, 0]], y=[X[-1, 1]], mode="markers", name="True end",
                                   marker=dict(size=12, symbol="diamond")))
    fig_state.add_trace(go.Scatter(x=[Xhat[0, 0]], y=[Xhat[0, 1]], mode="markers", name="Filtered start",
                                   marker=dict(size=10, symbol="circle")))
    fig_state.add_trace(go.Scatter(x=[Xhat[-1, 0]], y=[Xhat[-1, 1]], mode="markers", name="Filtered end",
                                   marker=dict(size=12, symbol="diamond")))

    # End confidence ellipse (filtered)
    end_mean = Xhat[-1, :2]
    end_cov = P_filt[-1, :2, :2]
    ex, ey = confidence_ellipse_points_2d(end_mean, end_cov, conf=ellipse_conf, num_points=250)
    fig_state.add_trace(
        go.Scatter(
            x=ex,
            y=ey,
            mode="lines",
            name=f"{int(ellipse_conf * 100)}% confidence ellipse (end)",
            line=dict(width=2, dash="dash"),
        )
    )

    fig_state.update_layout(
        title=dict(text="State-Space Trajectory: Price vs Drift", x=0.5),
        xaxis_title="Price level p",
        yaxis_title="Drift μ",
        template="plotly_white",
        height=600,
        width=750,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=80, b=120),
    )
    fig_state.write_image(f"{prefix}_state_space.png", scale=3)


    print("Saved PNG files:")
    print(f"  {prefix}_price.png")
    print(f"  {prefix}_drift.png")
    print(f"  {prefix}_state_space.png")

def save_mse_plot(mse, prefix="kalman_2d"):
    t = np.arange(len(mse))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=mse, mode="lines", name="Empirical MSE"
    ))

    fig.update_layout(
        title=dict(text="Monte Carlo Estimate of the Filter MSE", x=0.5),
        xaxis_title="Time index $n$",
        yaxis_title=r"$\mathrm{MSE}_n \approx \frac{1}{M}\sum_{i=1}^M \|X_n^{(i)}-\hat X_{n|n}^{(i)}\|^2$",
        template="plotly_white",
        height=500,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=140)
    )

    fig.write_image(f"{prefix}_mse.png", scale=3)
    print(f"Saved MSE PNG: {prefix}_mse.png")


def save_forecast_plot(A, H, Q, R, Xhat, P_filt, Y, horizon=20, prefix="kalman_2d", conf=0.95):
    """
    Forecast future observations using the filtered terminal state x_{N|N} and covariance P_{N|N}.
    Plot: observed y_n, filtered price estimate, forecast mean + confidence band.
    """
    import numpy as np
    import plotly.graph_objects as go

    A = np.atleast_2d(A)
    H = np.atleast_2d(H)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    N = Y.shape[0]  # observations: n=1..N
    t_obs = np.arange(1, N + 1)
    t_state = np.arange(0, N + 1)
    t_fore = np.arange(N + 1, N + horizon + 1)

    xN = Xhat[-1].reshape(-1, 1)   # x_{N|N}
    PN = P_filt[-1]               # P_{N|N}

    # z-score for normal CI
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(conf, 1.96)

    y_mean = np.zeros(horizon)
    y_std = np.zeros(horizon)

    A_h = np.eye(A.shape[0])          # A^h
    A_i = np.eye(A.shape[0])          # A^i for noise sum
    noise_sum = np.zeros_like(PN)     # sum_{i=0}^{h-1} A^i Q (A^i)^T

    for h in range(1, horizon + 1):
        # Update noise sum with term for i=h-1
        if h == 1:
            A_i = np.eye(A.shape[0])  # A^0
        else:
            A_i = A @ A_i             # now A^{h-1}
        noise_sum = noise_sum + A_i @ Q @ A_i.T

        # Update A^h
        A_h = A @ A_h

        # Predicted state moments
        x_pred = A_h @ xN
        P_pred = A_h @ PN @ A_h.T + noise_sum

        # Predicted observation moments
        y_pred = (H @ x_pred).item()
        y_var = (H @ P_pred @ H.T + R).item()

        y_mean[h - 1] = y_pred
        y_std[h - 1] = np.sqrt(max(y_var, 0.0))

    upper = y_mean + z * y_std
    lower = y_mean - z * y_std

    fig = go.Figure()

    # observed
    fig.add_trace(go.Scatter(
        x=t_obs, y=Y[:, 0], mode="lines", name="Observed price y_n"
    ))

    # filtered latent price estimate
    fig.add_trace(go.Scatter(
        x=t_state, y=Xhat[:, 0], mode="lines", name="Filtered price p̂_{n|n}"
    ))

    # forecast mean
    fig.add_trace(go.Scatter(
        x=t_fore, y=y_mean, mode="lines", name="Forecast mean E[y_{n}|Y_{1:N}]"
    ))

    # confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_fore, t_fore[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        mode="lines",
        line=dict(width=0),
        name=f"{int(conf * 100)}% forecast band"
    ))

    fig.update_layout(
        title=dict(text="Price Forecast using Estimated Drift (Kalman Filter)", x=0.5),
        xaxis_title="Time index n",
        yaxis_title="Price level",
        template="plotly_white",
        height=550,
        width=950,
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=80, b=120),
    )

    fig.write_image(f"{prefix}_forecast.png", scale=3)
    print(f"Saved forecast PNG: {prefix}_forecast.png")




# -----------------------------
# Main
# -----------------------------
def main():
    seed = 1 # int(np.random.randint(0, 128)) #20, 123 good
    rng = np.random.default_rng(seed)
    print(f"seed: {seed}")

    # Model (2D state: [price, drift])
    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    sigma_p = 0.10 #0.40
    sigma_mu = 0.02 #0.05
    Q = np.array([[sigma_p**2, 0.0],
                  [0.0, sigma_mu**2]])

    sigma_y = 0.8 #1.50
    R = np.array([[sigma_y**2]])

    x0_mean = np.array([0.0, 0.05])
    P0 = np.array([[1.0, 0.0],
               [0.0, 0.05]])
    P0 = np.array([[0.5, 0.0],
               [0.0, 0.08]])

    N = 50

    X, Y = simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng)
    Xhat, P_filt = kalman_filter(A, H, Q, R, x0_hat=x0_mean, P0=P0, Y=Y)

    save_separate_plots(X, Y, Xhat, P_filt, prefix="kalman_2d", ellipse_conf=0.95)

    fig = make_dashboard_plot(
        X, Y, Xhat, P_filt,
        out_html_path="kalman_2d_dashboard.html",
        out_png_prefix=None,
        ellipse_conf=0.95,
    )
    fig.show()
    # --- Monte Carlo MSE ---
    M = 5000  # e.g. 100-1000 depending on speed
    mse_seed = 12345  # fix for reproducibility of the MSE curve
    mse = empirical_mse_mc(A, H, Q, R, x0_mean, P0, N, M, seed=mse_seed)
    save_mse_plot(mse, prefix="kalman_2d")

    save_forecast_plot(A, H, Q, R, Xhat, P_filt, Y, horizon=20, prefix="kalman_2d", conf=0.95)


if __name__ == "__main__":
    main()
