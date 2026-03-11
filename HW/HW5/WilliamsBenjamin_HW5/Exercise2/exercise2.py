import numpy as np
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("vehicles.csv")

req = ["mpg", "wt", "am", "qsec", "drat", "gear"]
missing = [c for c in req if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in vehicles.csv: {missing}")

x_name = "wt"
y_name = "qsec"
z_name = "drat"
c_name = "mpg"
s_name = "gear"
shape_name = "am"

x = df[x_name].to_numpy(dtype=float)
y2 = df[y_name].to_numpy(dtype=float)
z = df[z_name].to_numpy(dtype=float)
color = df[c_name].to_numpy(dtype=float)

sz = df[s_name].to_numpy(dtype=float)
mn, mx = float(np.nanmin(sz)), float(np.nanmax(sz))
ms = 8 + 18 * (sz - mn) / (mx - mn + 1e-12)

am_vals = df[shape_name].to_numpy()
symbols = np.where(am_vals == 0, "circle", "square")

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x,
            y=y2,
            z=z,
            mode="markers",
            marker=dict(
                size=ms,
                color=color,
                symbol=symbols,
                colorscale="Viridis",
                showscale=True,
                opacity=1.0,
                line=dict(width=0)
            )
        )
    ]
)

fig.update_layout(
    title=f"6D Plot (x={x_name}, y={y_name}, z={z_name}, color={c_name}, size={s_name}, shape={shape_name})",
    scene=dict(
        xaxis=dict(title=x_name),
        yaxis=dict(title=y_name),
        zaxis=dict(title=z_name)
    )
)

fig.write_html("HW5_Exercise2_6DPlot.html", auto_open=True)