# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import socket

# ===============================
# 2. CREATE SYNTHETIC 10-YEARS DATASET
# ===============================
np.random.seed(42)

dates = pd.date_range(start="2013-01-01", end="2023-12-01", freq="MS")
products = ["A", "B", "C"]
categories = ["X", "Y", "Z"]

data_list = []
for date in dates:
    for prod in products:
        cat = np.random.choice(categories)
        base = 2000 + (date.month*50) + np.random.randint(-500,500)
        data_list.append([date, prod, cat, base])

df = pd.DataFrame(data_list, columns=["date","product","category","invoice_value"])

# Extract features
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Encode category
cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category"])

# ===============================
# 3. CREATE LAG FEATURES
# ===============================
df = df.sort_values(["category","date"])
df["lag1"] = df.groupby("category")["invoice_value"].shift(1)
df["lag2"] = df.groupby("category")["invoice_value"].shift(2)
df["lag3"] = df.groupby("category")["invoice_value"].shift(3)
df = df.bfill()  # backfill initial NaNs

# Features and target
features = ["month","year","category_encoded","lag1","lag2","lag3"]
target = "invoice_value"

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1,1)

# Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# ===============================
# 4. DEFINE TCN MODEL
# ===============================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]  # remove extra padding
        x = self.bn(x)
        x = self.relu(x)
        return x

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TCNModel, self).__init__()
        self.tcn1 = TCNBlock(input_size, hidden_size, kernel_size=2, dilation=1)
        self.tcn2 = TCNBlock(hidden_size, hidden_size, kernel_size=2, dilation=2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len=1, features)
        x = x.permute(0,2,1)  # (batch, features, seq_len)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:, :, -1]  # take last time step
        x = self.fc(x)
        return x

input_size = X_tensor.shape[1]
model = TCNModel(input_size)

# ===============================
# 5. TRAIN TCN MODEL
# ===============================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 500

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor.unsqueeze(1))  # add seq_len dimension
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1)%50 ==0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# ===============================
# 6. EVALUATE MODEL
# ===============================
model.eval()
pred_scaled = model(X_tensor.unsqueeze(1)).detach().numpy()
pred = scaler_y.inverse_transform(pred_scaled)

rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)
print(f"Training RMSE: {rmse:.2f}")
print(f"Training R2 Score: {r2*100:.2f}%")

# ===============================
# 7. FUTURE PREDICTIONS
# ===============================
future_dates = pd.date_range(start=df["date"].max()+pd.DateOffset(months=1), periods=12, freq="MS")
future_predictions = []

for cat in df["category"].unique():
    cat_id = cat_encoder.transform([cat])[0]
    df_cat = df[df["category"]==cat].sort_values("date")

    lag1, lag2, lag3 = df_cat.iloc[-1]["invoice_value"], df_cat.iloc[-2]["invoice_value"], df_cat.iloc[-3]["invoice_value"]

    preds_list = []
    for i, date in enumerate(future_dates):
        X_pred = np.array([[date.month, date.year, cat_id, lag1, lag2, lag3]], dtype=np.float32)
        X_pred_scaled = scaler_X.transform(X_pred)
        pred_scaled = model(torch.tensor(X_pred_scaled).unsqueeze(1)).detach().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        preds_list.append(pred)
        lag3, lag2, lag1 = lag2, lag1, pred

    temp = pd.DataFrame({
        "date": future_dates,
        "category":[cat]*len(future_dates),
        "predicted_invoice": preds_list
    })
    future_predictions.append(temp)

future_df_all = pd.concat(future_predictions)

# ===============================
# 8. DASH APP
# ===============================
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Product Category Wise Invoice Prediction (TCN)", style={"textAlign":"center"}),
    dcc.Dropdown(
        id="category_dropdown",
        options=[{"label":c,"value":c} for c in df["category"].unique()],
        value=df["category"].unique()[0]
    ),
    dcc.Graph(id="prediction_graph")
])

@app.callback(
    Output("prediction_graph","figure"),
    Input("category_dropdown","value")
)
def update_graph(selected_category):
    df_filtered = future_df_all[future_df_all["category"]==selected_category]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=df_filtered["predicted_invoice"],
        mode="lines+markers",
        name="Predicted Invoice"
    ))
    fig.update_layout(
        title=f"Predicted Invoice for {selected_category}",
        xaxis_title="Month",
        yaxis_title="Invoice Value",
        template="plotly_dark"
    )
    return fig

# ===============================
# 9. RUN DASH APP AUTO-PICK PORT
# ===============================
if __name__=="__main__":
    port = 8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                break
            port += 1
    print(f"Running Dash app on http://127.0.0.1:{port}")
    app.run_server(debug=True, port=port)
