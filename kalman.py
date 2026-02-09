import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def make_dashboard_plot(X, Y, Xhat, out_html_path="kalman_2d_dashboard.html", out_png_prefix=None):
    """
    Creates one Plotly dashboard (3 panels) and saves it as an HTML file.
    Optionally saves PNGs if out_png_prefix is provided AND kaleido is installed.
    """
    N = Y.shape[0]
    t = np.arange(N + 1)
    t_obs = np.arange(1, N + 1)

    # Build a 3-row dashboard:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=(
            "Price: latent vs observed vs filtered",
            "Drift (trend): true vs filtered",
            "State space: (price, drift) trajectory"
        )
    )

    # --- Row 1: price time series ---
    fig.add_trace(
        go.Scatter(x=t, y=X[:, 0], mode="lines", name="True latent price p_n"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_obs, y=Y[:, 0], mode="lines", name="Observed price y_n"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=Xhat[:, 0], mode="lines", name="Filtered price estimate"),
        row=1, col=1
    )
    fig.update_xaxes(title_text="n", row=1, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)

    # --- Row 2: drift time series ---
    fig.add_trace(
        go.Scatter(x=t, y=X[:, 1], mode="lines", name="True drift μ_n"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=Xhat[:, 1], mode="lines", name="Filtered drift estimate"),
        row=2, col=1
    )
    fig.update_xaxes(title_text="n", row=2, col=1)
    fig.update_yaxes(title_text="drift μ", row=2, col=1)

    # --- Row 3: state-space trajectory ---
    # fig.add_trace(
    #     go.Scatter(x=X[:, 0], y=X[:, 1], mode="lines", name="True trajectory (p, μ)"),
    #     row=3, col=1
    # )
    # fig.add_trace(
    #     go.Scatter(x=Xhat[:, 0], y=Xhat[:, 1], mode="lines", name="Filtered trajectory (p, μ)"),
    #     row=3, col=1
    # )
    # fig.update_xaxes(title_text="price p", row=3, col=1)
    # fig.update_yaxes(title_text="drift μ", row=3, col=1)
    # --- Row 3: state-space trajectory with markers ---
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="lines+markers",
            name="True trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2)
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=Xhat[:, 0],
            y=Xhat[:, 1],
            mode="lines+markers",
            name="Filtered trajectory (p, μ)",
            marker=dict(size=5),
            line=dict(width=2)
        ),
        row=3, col=1
    )


    # Global styling (clean + readable)
    fig.update_layout(
        title="Kalman Filter (2D State Financial Model): Price + Drift",
        template="plotly_white",
        height=1000,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=90, b=60)
    )

    # Save dashboard
    fig.write_html(out_html_path, include_plotlyjs="cdn")
    print(f"Wrote interactive dashboard to: {out_html_path}")

    # Optional PNG export (requires kaleido)
    if out_png_prefix is not None:
        try:
            fig.write_image(f"{out_png_prefix}_dashboard.png", scale=2)
            print(f"Wrote PNG to: {out_png_prefix}_dashboard.png")
        except Exception as e:
            print("PNG export failed (likely missing 'kaleido').")
            print("Install it via: pip install -U kaleido")
            print(f"Error was: {e}")

    return fig

def save_separate_plots(X, Y, Xhat, prefix="kalman_2d"):
    """
    Create and save three separate PNG figures:
    1) Price time series
    2) Drift (trend) time series
    3) State-space trajectory
    Requires: pip install -U kaleido
    """

    N = Y.shape[0]
    t = np.arange(N + 1)
    t_obs = np.arange(1, N + 1)

    # -----------------------------
    # 1) Price time series
    # -----------------------------
    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=t, y=X[:, 0],
        mode="lines",
        name="True latent price $p_n$"
    ))

    fig_price.add_trace(go.Scatter(
        x=t_obs, y=Y[:, 0],
        mode="lines",
        name="Observed price $y_n$"
    ))

    fig_price.add_trace(go.Scatter(
        x=t, y=Xhat[:, 0],
        mode="lines",
        name="Filtered estimate $\hat p_{n|n}$"
    ))

    fig_price.update_layout(
        title=dict(
            text="Latent Price Estimation via Kalman Filter",
            x=0.5
        ),
        xaxis_title="Time index $n$",
        yaxis_title="Price level",
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
        margin=dict(t=80, b=120)
    )

    fig_price.write_image(f"{prefix}_price.png", scale=3)

    # -----------------------------
    # 2) Drift (trend) time series
    # -----------------------------
    fig_drift = go.Figure()

    fig_drift.add_trace(go.Scatter(
        x=t, y=X[:, 1],
        mode="lines",
        name="True drift $\\mu_n$"
    ))

    fig_drift.add_trace(go.Scatter(
        x=t, y=Xhat[:, 1],
        mode="lines",
        name="Filtered drift estimate $\\hat\\mu_{n|n}$"
    ))

    fig_drift.update_layout(
        title=dict(
            text="Trend (Drift) Estimation via Kalman Filter",
            x=0.5
        ),
        xaxis_title="Time index $n$",
        yaxis_title="Drift $\\mu$",
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
        margin=dict(t=80, b=120)
    )

    fig_drift.write_image(f"{prefix}_drift.png", scale=3)

    # -----------------------------
    # 3) State-space trajectory
    # -----------------------------
    fig_state = go.Figure()

    fig_state.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode="lines+markers",
        name="True trajectory $(p_n, \\mu_n)$",
        marker=dict(size=5),
        line=dict(width=2)
    ))

    fig_state.add_trace(go.Scatter(
        x=Xhat[:, 0],
        y=Xhat[:, 1],
        mode="lines+markers",
        name="Filtered trajectory $(\\hat p_{n|n}, \\hat\\mu_{n|n})$",
        marker=dict(size=5),
        line=dict(width=2)
    ))

    fig_state.update_layout(
        title=dict(
            text="State-Space Trajectory: Price vs Drift",
            x=0.5
        ),
        xaxis_title="Price level $p$",
        yaxis_title="Drift $\\mu$",
        template="plotly_white",
        height=600,
        width=700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=120)
    )

    fig_state.write_image(f"{prefix}_state_space.png", scale=3)

    print("Saved PNG files:")
    print(f"  {prefix}_price.png")
    print(f"  {prefix}_drift.png")
    print(f"  {prefix}_state_space.png")



def main():
    rng = np.random.default_rng(np.random.randint(0, 128))
    print(f"rng: {rng}")

    A = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    sigma_p = 0.40
    sigma_mu = 0.05
    Q = np.array([[sigma_p**2, 0.0],
                  [0.0, sigma_mu**2]])

    sigma_y = 0.80
    R = np.array([[sigma_y**2]])

    x0_mean = np.array([0.0, 0.05])
    P0 = np.array([[2.0, 0.0],
                   [0.0, 0.2]])

    N = 120

    X, Y = simulate_lgssm_2d(A, H, Q, R, x0_mean, P0, N, rng)
    Xhat, _ = kalman_filter(A, H, Q, R, x0_hat=x0_mean, P0=P0, Y=Y)

    # One beautiful dashboard you can view all at once + saved to file:
    save_separate_plots(X, Y, Xhat, prefix="kalman_2d")


    # In notebooks: display inline
    # fig.show()
    # In scripts: it will open in browser if you uncomment:

if __name__ == "__main__":
    main()
