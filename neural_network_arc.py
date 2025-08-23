# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import socket
from tqdm import trange

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
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category"])

# ===============================
# 3. CREATE LAG FEATURES
# ===============================
df = df.sort_values(["category","date"])
df["lag1"] = df.groupby("category")["invoice_value"].shift(1)
df["lag2"] = df.groupby("category")["invoice_value"].shift(2)
df["lag3"] = df.groupby("category")["invoice_value"].shift(3)
df = df.bfill()

features = ["month","year","category_encoded","lag1","lag2","lag3"]
target = "invoice_value"

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1,1)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# ===============================
# 4. DEFINE MODELS
# ===============================
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self,x):
        return self.model(x)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super(TCNBlock,self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self,x):
        x = self.conv(x)
        x = x[:,:, :-self.conv.padding[0]]  # remove extra padding
        x = self.bn(x)
        x = self.relu(x)
        return x

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TCNModel,self).__init__()
        self.tcn1 = TCNBlock(input_size, hidden_size, kernel_size=2, dilation=1)
        self.tcn2 = TCNBlock(hidden_size, hidden_size, kernel_size=2, dilation=2)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        x = x.permute(0,2,1)  # batch, features, seq_len
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = x[:,:, -1]
        x = self.fc(x)
        return x

# ===============================
# 5. TRAIN FUNCTION WITH PROGRESS BAR
# ===============================
def train_nn(model, X, y, epochs=500, lr=0.01, use_seq=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in trange(epochs, desc="Training"):
        optimizer.zero_grad()
        if use_seq:
            outputs = model(X.unsqueeze(1))
        else:
            outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    return model

# ===============================
# 6. TRAIN MLP & TCN
# ===============================
input_size = X_tensor.shape[1]

mlp_model = MLPModel(input_size)
tcn_model = TCNModel(input_size)

print("Training MLP model:")
mlp_model = train_nn(mlp_model, X_tensor, y_tensor, epochs=300)

print("Training TCN model:")
tcn_model = train_nn(tcn_model, X_tensor, y_tensor, epochs=300, use_seq=True)

# ===============================
# 7. EVALUATE MODELS
# ===============================
mlp_pred = scaler_y.inverse_transform(mlp_model(X_tensor).detach().numpy())
tcn_pred = scaler_y.inverse_transform(tcn_model(X_tensor.unsqueeze(1)).detach().numpy())
y_actual = y

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return rmse,r2,mae

mlp_rmse, mlp_r2, mlp_mae = eval_metrics(y_actual, mlp_pred)
tcn_rmse, tcn_r2, tcn_mae = eval_metrics(y_actual, tcn_pred)

print(f"MLP => RMSE: {mlp_rmse:.2f}, R2: {mlp_r2*100:.2f}%, MAE: {mlp_mae:.2f}")
print(f"TCN => RMSE: {tcn_rmse:.2f}, R2: {tcn_r2*100:.2f}%, MAE: {tcn_mae:.2f}")

best_model = mlp_model if mlp_rmse < tcn_rmse else tcn_model
best_name = "MLP" if mlp_rmse < tcn_rmse else "TCN"
use_seq = False if best_name=="MLP" else True
print("Best Model Selected:", best_name)

# ===============================
# 8. BOOSTING ON BEST MODEL
# ===============================
for i in range(3):
    best_model = train_nn(best_model, X_tensor, y_tensor, epochs=150, use_seq=use_seq)
print("Boosting complete.")

# ===============================
# 9. FUTURE PREDICTIONS
# ===============================
future_dates = pd.date_range(start=df["date"].max()+pd.DateOffset(months=1), periods=12, freq="MS")
future_predictions = []

best_model.eval()
with torch.no_grad():
    for cat in df["category"].unique():
        cat_id = cat_encoder.transform([cat])[0]
        df_cat = df[df["category"]==cat].sort_values("date")
        lag1, lag2, lag3 = df_cat.iloc[-1]["invoice_value"], df_cat.iloc[-2]["invoice_value"], df_cat.iloc[-3]["invoice_value"]

        preds_list = []
        for date in future_dates:
            X_pred = np.array([[date.month, date.year, cat_id, lag1, lag2, lag3]], dtype=np.float32)
            X_pred_scaled = scaler_X.transform(X_pred)
            X_tensor_pred = torch.tensor(X_pred_scaled, dtype=torch.float32)

            if use_seq:  # TCN
                pred_scaled = best_model(X_tensor_pred.unsqueeze(0).unsqueeze(1))
            else:        # MLP
                pred_scaled = best_model(X_tensor_pred.unsqueeze(0))

            # Fix shape for inverse transform
            pred_scaled = pred_scaled.squeeze().detach().numpy().reshape(-1,1)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]

            preds_list.append(pred)
            lag3, lag2, lag1 = lag2, lag1, pred

        temp = pd.DataFrame({
            "date": future_dates,
            "category": [cat]*len(future_dates),
            "predicted_invoice": preds_list
        })
        future_predictions.append(temp)

future_df_all = pd.concat(future_predictions)

# ===============================
# 10. DASH APP
# ===============================
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Future Invoice Prediction - Boosted Neural Model", style={"textAlign":"center"}),
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
        x=df_filtered["date"], y=df_filtered["predicted_invoice"],
        mode="lines+markers", name="Predicted Invoice"
    ))
    fig.update_layout(
        title=f"Predicted Invoice for {selected_category} ({best_name} Boosted)",
        xaxis_title="Month", yaxis_title="Invoice Value",
        template="plotly_dark"
    )
    return fig

# ===============================
# 11. RUN DASH APP AUTO-PICK PORT
# ===============================
if __name__=="__main__":
    port=8050
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1",port))!=0:
                break
            port+=1
    print(f"Running Dash app on http://127.0.0.1:{port}")
    app.run_server(debug=True, port=port)
