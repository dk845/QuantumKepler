import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_transit_simulation():
    # Orbital parameters
    n_frames = 120
    orbit_radius = 0.7
    planet_radius = 0.12
    star_radius = 0.25
    
    times = np.linspace(0, 4 * np.pi, n_frames)
    
    # Planet position
    planet_x = orbit_radius * np.cos(times)
    planet_y = orbit_radius * np.sin(times) * 0.3  # elliptical orbit view
    planet_z = orbit_radius * np.sin(times)  # depth (z axis)
    
    # Compute flux dip — only when planet is in front of star (z > 0) and overlapping
    def compute_flux(px, py, pz):
        dist = np.sqrt(px**2 + py**2)
        in_front = pz > 0  # planet is in front of star
        overlap = dist < (star_radius + planet_radius)
        if in_front and overlap:
            # How much overlap
            overlap_depth = max(0, (star_radius + planet_radius - dist))
            dip = (planet_radius / star_radius) ** 2 * min(1, overlap_depth / (2 * planet_radius))
            return 1.0 - dip
        return 1.0

    flux_values = [compute_flux(planet_x[i], planet_y[i], planet_z[i]) for i in range(n_frames)]

    # Star glow points
    theta_star = np.linspace(0, 2 * np.pi, 100)
    star_cx = star_radius * np.cos(theta_star)
    star_cy = star_radius * np.sin(theta_star)

    # Build frames
    frames = []
    flux_so_far_x = []
    flux_so_far_y = []

    for i in range(n_frames):
        flux_so_far_x.append(i)
        flux_so_far_y.append(flux_values[i])

        # Planet circle
        theta_p = np.linspace(0, 2 * np.pi, 60)
        px_circle = planet_x[i] + planet_radius * np.cos(theta_p)
        py_circle = planet_y[i] + planet_radius * np.sin(theta_p)

        # Star brightness
        is_transit = flux_values[i] < 0.99
        star_color = "#FF6A00" if not is_transit else "#FF3300"
        star_opacity = 1.0 if not is_transit else 0.7

        frame = go.Frame(
            data=[
                # Star
                go.Scatter(
                    x=star_cx, y=star_cy,
                    mode="lines",
                    fill="toself",
                    fillcolor=star_color,
                    line=dict(color=star_color),
                    opacity=star_opacity,
                    name="Star"
                ),
                # Orbit path
                go.Scatter(
                    x=orbit_radius * np.cos(np.linspace(0, 2*np.pi, 200)),
                    y=orbit_radius * np.sin(np.linspace(0, 2*np.pi, 200)) * 0.3,
                    mode="lines",
                    line=dict(color="#333333", dash="dot", width=1),
                    name="Orbit"
                ),
                # Planet
                go.Scatter(
                    x=px_circle, y=py_circle,
                    mode="lines",
                    fill="toself",
                    fillcolor="#1E90FF",
                    line=dict(color="#00BFFF"),
                    name="Planet"
                ),
                # Light curve so far
                go.Scatter(
                    x=flux_so_far_x, y=flux_so_far_y,
                    mode="lines",
                    line=dict(color="#A855F7", width=2.5),
                    name="Flux",
                    xaxis="x2", yaxis="y2"
                ),
                # Current point on light curve
                go.Scatter(
                    x=[flux_so_far_x[-1]], y=[flux_so_far_y[-1]],
                    mode="markers",
                    marker=dict(color="#FFFFFF", size=10, symbol="circle"),
                    name="Current",
                    xaxis="x2", yaxis="y2"
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    # Initial data
    fig = go.Figure(
        data=[
            go.Scatter(x=star_cx, y=star_cy, mode="lines", fill="toself",
                      fillcolor="#FF6A00", line=dict(color="#FF6A00"), name="Star"),
            go.Scatter(
                x=orbit_radius * np.cos(np.linspace(0, 2*np.pi, 200)),
                y=orbit_radius * np.sin(np.linspace(0, 2*np.pi, 200)) * 0.3,
                mode="lines", line=dict(color="#333333", dash="dot", width=1), name="Orbit"),
            go.Scatter(
                x=[planet_x[0] + planet_radius * np.cos(t) for t in np.linspace(0, 2*np.pi, 60)],
                y=[planet_y[0] + planet_radius * np.sin(t) for t in np.linspace(0, 2*np.pi, 60)],
                mode="lines", fill="toself", fillcolor="#1E90FF",
                line=dict(color="#00BFFF"), name="Planet"),
            go.Scatter(x=[0], y=[1.0], mode="lines",
                      line=dict(color="#A855F7", width=2.5), name="Flux",
                      xaxis="x2", yaxis="y2"),
            go.Scatter(x=[0], y=[1.0], mode="markers",
                      marker=dict(color="white", size=10), name="Current",
                      xaxis="x2", yaxis="y2")
        ],
        frames=frames
    )

    fig.update_layout(
        title=dict(
            text="🪐 Exoplanet Transit Simulation — Watch the Light Dim!",
            font=dict(size=20, color="white")
        ),
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        xaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                  showticklabels=False, domain=[0, 1]),
        yaxis=dict(range=[-0.6, 0.6], showgrid=False, zeroline=False,
                  showticklabels=False, domain=[0.35, 1]),
        xaxis2=dict(range=[0, n_frames], showgrid=True, gridcolor="#222",
                   color="white", title="Time (frames)", domain=[0, 1]),
        yaxis2=dict(range=[0.95, 1.02], showgrid=True, gridcolor="#222",
                   color="white", title="Normalized Flux", anchor="x2"),
        showlegend=False,
        height=750,
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "y": 0.32,
            "x": 0.5,
            "xanchor": "center",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 60, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                }
            ]
        }],
        sliders=[{
            "steps": [{"args": [[f.name], {"frame": {"duration": 60, "redraw": True},
                                           "mode": "immediate"}],
                       "method": "animate", "label": ""} for f in frames],
            "x": 0.1, "len": 0.8, "y": 0.28,
            "currentvalue": {"prefix": "Frame: ", "font": {"color": "white"}},
            "bgcolor": "#333", "bordercolor": "#666"
        }]
    )

    import os
    os.makedirs("visualizations/output", exist_ok=True)
    out = "visualizations/output/transit_simulation.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()

if __name__ == "__main__":
    generate_transit_simulation()