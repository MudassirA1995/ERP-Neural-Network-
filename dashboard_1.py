import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import webbrowser
from threading import Timer

# ------------------ Dummy Data ------------------ #
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=30)
products = ["Alpha", "Beta", "Gamma", "Delta"]
categories = ["Electronics", "Clothing", "Home", "Sports"]
regions = ["North", "South", "East", "West"]

# Current prediction dataset
df = pd.DataFrame({
    "Live Product": np.random.choice(products, 100),
    "Product Category": np.random.choice(categories, 100),
    "Date": np.random.choice(dates, 100),
    "Predicted Invoice Value": np.random.randint(1000, 10000, 100)
})

# Previous dataset (for doughnut)
prev_df = pd.DataFrame({
    "Product Category": np.random.choice(categories, 80),
    "Region": np.random.choice(regions, 80),
    "Prev Invoice Value": np.random.randint(500, 8000, 80)
})

# ------------------ App Init ------------------ #
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# ------------------ Animated Text Style ------------------ #
animated_text_style = {
    "color": "#88ddff",
    "fontSize": "14px",
    "animation": "pulse 5s infinite",
    "textAlign": "justify"
}

# Custom CSS animations
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Predictive Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        </style>
    </head>
    <body style="background-color: #000;">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

# ------------------ Layout ------------------ #
card_style = {"backgroundColor": "#000", "padding": "20px", "borderRadius": "20px", "boxShadow": "0px 0px 25px #00f0ff", "height": "100%"}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1("Predictive Analysis Dashboard", 
                    style={"color": "#00f0ff", "textAlign": "right", "fontFamily": "Orbitron", "marginBottom": "20px", "fontSize": "38px", "textShadow": "0px 0px 15px #00f0ff"})
        )
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("📊 Category-Wise Trends", style={"color": "#00f0ff"}),
                html.P("This graph showcases how predicted invoice values evolve across different dates, filtered by product category. It helps understand seasonal fluctuations, demand cycles, and long-term category performance.", style=animated_text_style),
                html.Label("Select Product Category", style={"color": "#00f0ff"}),
                dcc.Dropdown(
                    id="category-filter",
                    options=[{"label": c, "value": c} for c in categories],
                    value=categories[0],
                    style={"backgroundColor": "#111", "color": "#00f0ff"}
                ),
                html.Div([
                    dbc.Button("Line Chart", id="cat-line-btn", color="primary", outline=True, style={"marginRight": "5px"}),
                    dbc.Button("Bar Chart", id="cat-bar-btn", color="primary", outline=True)
                ], style={"marginTop": "10px", "marginBottom": "10px"}),
                dcc.Graph(id="graph1")
            ], style=card_style)
        ], width=6),

        dbc.Col([
            html.Div([
                html.H4("📈 Product-Specific Insights", style={"color": "#00f0ff"}),
                html.P("Analyze product-level performance trends. This visualization reveals how each product's invoice values change across dates, helping identify market winners and underperformers.", style=animated_text_style),
                html.Label("Select Product", style={"color": "#00f0ff"}),
                dcc.Dropdown(
                    id="product-filter",
                    options=[{"label": p, "value": p} for p in products],
                    value=products[0],
                    style={"backgroundColor": "#111", "color": "#00f0ff"}
                ),
                html.Div([
                    dbc.Button("Line Chart", id="prod-line-btn", color="primary", outline=True, style={"marginRight": "5px"}),
                    dbc.Button("Bar Chart", id="prod-bar-btn", color="primary", outline=True)
                ], style={"marginTop": "10px", "marginBottom": "10px"}),
                dcc.Graph(id="graph2")
            ], style=card_style)
        ], width=6),
    ], style={"marginBottom": "30px", "height": "500px"}),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("🌍 Regional Distribution by Category", style={"color": "#00f0ff"}),
                html.P("The doughnut chart illustrates the proportional contribution of regions to the selected product category's invoice values. It helps stakeholders understand geographic dominance and gaps.", style=animated_text_style),
                html.Label("Select Product Category", style={"color": "#00f0ff"}),
                dcc.Dropdown(
                    id="region-category-filter",
                    options=[{"label": c, "value": c} for c in categories],
                    value=categories[0],
                    style={"backgroundColor": "#111", "color": "#00f0ff"}
                ),
                dcc.Graph(id="graph3")
            ], style=card_style)
        ], width=6),

        dbc.Col([
            html.Div([
                html.H4("🔵 Bubble Chart Analysis", style={"color": "#00f0ff"}),
                html.P("The bubble chart highlights the overall predicted invoice values by category. Larger bubbles indicate higher predicted revenue, offering a visual hierarchy of business impact.", style=animated_text_style),
                dcc.Graph(id="graph4")
            ], style=card_style)
        ], width=6),
    ], style={"height": "500px"})
], fluid=True, style={"backgroundColor": "#000", "padding": "20px"})

# ------------------ Callbacks ------------------ #
@app.callback(
    Output("graph1", "figure"),
    [Input("category-filter", "value"), Input("cat-line-btn", "n_clicks"), Input("cat-bar-btn", "n_clicks")]
)
def update_graph1(selected_category, line_click, bar_click):
    filtered = df[df["Product Category"] == selected_category]
    chart_type = "line"
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("cat-bar-btn"):
        chart_type = "bar"
    if chart_type == "line":
        fig = px.line(filtered, x="Date", y="Predicted Invoice Value", color="Product Category", markers=True)
    else:
        fig = px.bar(filtered, x="Date", y="Predicted Invoice Value", color="Product Category")
    fig.update_layout(template="plotly_dark", title="Invoice Value by Date (Category Filter)", title_font_color="#00f0ff", font_color="#00f0ff", plot_bgcolor="#111", paper_bgcolor="#111")
    return fig

@app.callback(
    Output("graph2", "figure"),
    [Input("product-filter", "value"), Input("prod-line-btn", "n_clicks"), Input("prod-bar-btn", "n_clicks")]
)
def update_graph2(selected_product, line_click, bar_click):
    filtered = df[df["Live Product"] == selected_product]
    chart_type = "line"
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("prod-bar-btn"):
        chart_type = "bar"
    if chart_type == "line":
        fig = px.line(filtered, x="Date", y="Predicted Invoice Value", color="Live Product", markers=True)
    else:
        fig = px.bar(filtered, x="Date", y="Predicted Invoice Value", color="Live Product")
    fig.update_layout(template="plotly_dark", title="Invoice Value by Date (Product Filter)", title_font_color="#00f0ff", font_color="#00f0ff", plot_bgcolor="#111", paper_bgcolor="#111")
    return fig

@app.callback(
    Output("graph3", "figure"),
    Input("region-category-filter", "value")
)
def update_graph3(selected_category):
    filtered = prev_df[prev_df["Product Category"] == selected_category]
    fig = px.pie(filtered, values="Prev Invoice Value", names="Region", hole=0.5,
                 title=f"Regional Trend for {selected_category}")
    fig.update_traces(textposition="inside", textinfo="percent+label", marker=dict(colors=["#00f0ff", "#00aaff", "#0077aa", "#004477"]))
    fig.update_layout(template="plotly_dark", title_font_color="#00f0ff", font_color="#00f0ff", plot_bgcolor="#111", paper_bgcolor="#111")
    return fig

@app.callback(
    Output("graph4", "figure"),
    Input("category-filter", "value")
)
def update_graph4(_):
    agg = df.groupby("Product Category").agg({"Predicted Invoice Value": "sum"}).reset_index()
    agg["Size"] = agg["Predicted Invoice Value"] / 200
    fig = px.scatter(agg, x="Product Category", y="Predicted Invoice Value", size="Size", color="Product Category",
                     title="Bubble Chart: Product Category Invoice Value")
    fig.update_layout(template="plotly_dark", title_font_color="#00f0ff", font_color="#00f0ff", plot_bgcolor="#111", paper_bgcolor="#111")
    return fig

# ------------------ Auto-open Browser ------------------ #
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=False)
