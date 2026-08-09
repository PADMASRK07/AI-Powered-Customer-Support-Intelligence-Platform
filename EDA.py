import os
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# =========================
# LOAD DATASET
# =========================

csv_path = "D:\Customer Support\Data set\customer_support_tickets_clean.csv"  # Update if needed

print("Current Working Directory:")
print(os.getcwd())

print("\nLooking for:")
print(csv_path)

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"Dataset not found: {csv_path}"
    )

df = pd.read_csv(csv_path)

print("\nDataset Loaded Successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# =========================
# COLUMN VALIDATION
# =========================

required_columns = [
    "Ticket Type",
    "Ticket Priority",
    "Ticket Channel",
    "Customer Satisfaction Rating",
    "Time to Resolution"
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing columns: {missing_cols}"
    )

# =========================
# KPIs
# =========================

total_tickets = len(df)

avg_resolution = round(
    df["Time to Resolution"].mean(),
    2
)

avg_rating = round(
    df["Customer Satisfaction Rating"].mean(),
    2
)

# =========================
# VISUALIZATIONS
# =========================

ticket_type_fig = px.histogram(
    df,
    x="Ticket Type",
    color="Ticket Type",
    title="Ticket Type Distribution"
)

priority_fig = px.histogram(
    df,
    x="Ticket Priority",
    color="Ticket Priority",
    title="Ticket Priority Distribution"
)

channel_fig = px.histogram(
    df,
    x="Ticket Channel",
    color="Ticket Channel",
    title="Channel Performance"
)

resolution_fig = px.histogram(
    df,
    x="Time to Resolution",
    nbins=30,
    title="Resolution Time Distribution"
)

satisfaction_fig = px.histogram(
    df,
    x="Customer Satisfaction Rating",
    color="Customer Satisfaction Rating",
    title="Customer Satisfaction Distribution"
)

# =========================
# HEATMAP
# =========================

heatmap_data = pd.crosstab(
    df["Ticket Channel"],
    df["Customer Satisfaction Rating"]
)

heatmap_fig = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    title="Channel vs Satisfaction Heatmap"
)

# =========================
# DASH APP
# =========================

app = dash.Dash(__name__)

app.layout = html.Div(

    style={
        "padding": "20px",
        "fontFamily": "Arial"
    },

    children=[

        html.H1(
            "Customer Support Intelligence Dashboard",
            style={"textAlign": "center"}
        ),

        html.Hr(),

        html.Div(

            style={
                "display": "flex",
                "justifyContent": "space-around",
                "marginBottom": "20px"
            },

            children=[

                html.Div([
                    html.H3("Total Tickets"),
                    html.H2(total_tickets)
                ]),

                html.Div([
                    html.H3("Avg Resolution Time"),
                    html.H2(avg_resolution)
                ]),

                html.Div([
                    html.H3("Avg Satisfaction"),
                    html.H2(avg_rating)
                ])
            ]
        ),

        dcc.Graph(figure=ticket_type_fig),

        dcc.Graph(figure=priority_fig),

        dcc.Graph(figure=channel_fig),

        dcc.Graph(figure=satisfaction_fig),

        dcc.Graph(figure=resolution_fig),

        dcc.Graph(figure=heatmap_fig)

    ]
)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    print("\nStarting Dash Server...")
    print("Open: http://127.0.0.1:8050")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050
    )