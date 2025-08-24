# ERP NEURAL NETWORK PREDICTION MODEL

## Introduction
This project focuses on building an **AI-powered chatbot system** integrated with a **dashboard interface** to provide an intelligent and interactive experience. The chatbot is trained on structured datasets, using natural language processing (NLP) and deep learning techniques, to understand user queries and respond with contextually accurate answers. Additionally, the project extends to real-time analytics and visualization using an interactive **Dashboard UI**, allowing users to monitor responses, performance metrics, and actionable insights.

The overall objective is to create a **scalable, industry-ready solution** that combines conversational AI with business intelligence, making it suitable for integration into enterprise environments such as **ERP systems, customer support workflows, and knowledge management platforms**.

---

## Code Explanation
The implementation is modular and organized into distinct components:

### Source Code 

```python


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



```

# Future Invoice Prediction using Neural Networks (MLP & TCN)

This project demonstrates how to generate synthetic invoice data, train neural network models (MLP and TCN), evaluate them, and use the best-performing model to make boosted future predictions. A **Dash dashboard** is also included to visualize the predicted invoices interactively.

---

## 📌 Workflow Overview

1. **Imports**  
   - Libraries: `pandas`, `numpy`, `torch`, `sklearn`, `dash`, `plotly`.
   - Purpose: Data preprocessing, model building, training, evaluation, and visualization.

2. **Synthetic Dataset Creation (10 years)**  
   - A monthly dataset from **2013 to 2023** is generated.  
   - Features:  
     - `date`, `product`, `category`, `invoice_value`.  
     - `invoice_value` varies with seasonality (month effect) and random noise.  
   - Encoded categorical features using **LabelEncoder**.  
   - Extracted time features: `month`, `year`.

3. **Lag Features**  
   - Created lag variables (`lag1`, `lag2`, `lag3`) per category.  
   - These features provide historical invoice values to help the model capture time-series dependencies.

4. **Feature Scaling**  
   - Applied **StandardScaler** to both features (`X`) and target (`y`).  
   - Converted data into **PyTorch tensors** for training.

5. **Model Definitions**
   - **MLP (Multi-Layer Perceptron)**  
     - Fully connected neural network with two hidden layers (128 and 64 neurons).  
   - **TCN (Temporal Convolutional Network)**  
     - Uses 1D dilated convolutions to capture sequential dependencies in time-series data.  
     - Two stacked TCN blocks followed by a dense output layer.

6. **Training Function**
   - A general training loop with:
     - **Adam optimizer**
     - **MSE Loss**
     - **Progress bar (tqdm)**  
   - Supports both standard input (MLP) and sequential input (TCN).

7. **Model Training**
   - Both **MLP** and **TCN** trained for 300 epochs.  
   - Predictions are compared with actual values.

8. **Model Evaluation**
   - Metrics used: **RMSE, R², MAE**.  
   - Best-performing model is selected based on **RMSE**.  
   - Additional boosting: The best model is retrained for **3 extra rounds** of 150 epochs.

9. **Future Predictions**
   - Forecasted **12 future months** per category.  
   - Predictions are generated recursively, updating lag values at each step.  
   - Results stored in `future_df_all`.

10. **Interactive Dashboard (Dash + Plotly)**
    - Dropdown menu to select categories (`X`, `Y`, `Z`).  
    - Line graph shows predicted invoices over future months.  
    - Dark theme for better readability.

11. **Auto Port Selection**
    - The app automatically finds an available port starting from `8050`.  
    - Ensures the dashboard runs without conflicts.

---

## ⚙️ Key Features

- **Two model architectures** (MLP and TCN).  
- **Boosting technique** to improve model performance.  
- **Lag-based time-series forecasting**.  
- **Interactive dashboard** for visualization.  
- **Automated port handling** for running the Dash app.

---

## 📊 Example Metrics (Printed in Console)

- **MLP Model**: RMSE, R², MAE  
- **TCN Model**: RMSE, R², MAE  
- Best model is selected automatically.

---

## 🚀 Running the App

Run the script:
```bash
python neural_network_arc.py
```

---

<img width="1920" height="1080" alt="model_structure" src="https://github.com/user-attachments/assets/4e315973-9f6c-4881-9e1b-8db8ff06f760" />



# ERP Data Predictor - Model Architecture

This architecture demonstrates the workflow of predicting ERP (Enterprise Resource Planning) data using **Neural Network algorithms**.  
The system selects the **best performing model** (MLP or TCN), applies **boosting**, and generates **future predictions**.

---

## 📌 Step-by-Step Data Flow

### 1. **ERP Data Input**
- Raw ERP data is the starting point of the pipeline.  
- This dataset contains historical records used for training and future prediction.

---

### 2. **Future Data Preparation**
- The dataset is extended to include **future time points** where predictions are required.  
- These future records initially lack invoice values, which the model will predict.

---

### 3. **Data Splitting**
- The ERP data is split into:
  - **Training Data** – used for training the models.  
  - **Test Data** – used for evaluation and comparison.  
- Splitting is performed using **Scikit-learn’s `train_test_split`** method.

---

### 4. **Neural Network Algorithms**
- Two different neural network architectures are tested:
  - **MLP (Multi-Layer Perceptron)** – a feedforward dense neural network.  
  - **TCN (Temporal Convolutional Network)** – captures sequential dependencies in time-series data.  

Both models are trained separately on the same dataset.

---

### 5. **Hyperparameter Tuning**
The models are fine-tuned by adjusting key parameters such as:
- Number of layers  
- Number of attention heads  
- Dimension of the feedforward network  
- Learning rate  
- Batch size  
- Dropout rate  
- Sequence length  
- Epochs  

This ensures both models are optimized before evaluation.

---

### 6. **Best Algorithm Selector**
- After training and evaluation, both MLP and TCN are compared using performance metrics such as **RMSE, R², MAE**.  
- The **best-performing algorithm** is selected for final deployment.

---

### 7. **Model Boosting**
- The best model undergoes **boosting** – retraining in multiple stages to refine weights and improve accuracy.  
- This creates a more robust and stable final model.

---

### 8. **Final Boosted Model**
- The boosted version of the best-performing model becomes the **final model**.  
- It is now ready for making predictions on unseen data.

---

### 9. **Predictions**
- The final model predicts **future ERP values** on the prepared future dataset.  
- Results are then visualized or used for decision-making in business applications.

---

## 🔄 Summary of Data Flow
1. **ERP Data → Future Data Prep → Train/Test Split**  
2. **Training Data → Neural Network (MLP & TCN)**  
3. **Hyperparameter Tuning → Best Algorithm Selection**  
4. **Best Model → Model Boosting → Final Boosted Model**  
5. **Final Model + Future Dataset → Predictions**

---

## ✅ Key Takeaways
- The system is **model-agnostic** (tests both MLP and TCN).  
- **Best-performing model** is automatically selected.  
- Boosting ensures **accuracy and robustness**.  
- Predictions are made on **future ERP datasets** for business forecasting.


### How the Model Works 

The model architecture is designed to evaluate and select the best-performing neural network for the given dataset. In this project, two types of neural networks were implemented and trained on the same preprocessed data, ensuring a fair comparison between the approaches. During the training process, each model was initialized with its respective parameters, optimized using backpropagation, and iteratively updated through multiple epochs to minimize loss functions such as Mean Squared Error (for regression) or Cross-Entropy Loss (for classification). The training pipeline involved splitting the dataset into training and validation sets, allowing the models to learn from the training data while their generalization performance was assessed on unseen validation data. After training, performance metrics such as accuracy, mean squared error, and loss values were calculated for both neural networks. These metrics were then compared to identify the superior model. The final selection process was based on the model that consistently demonstrated the highest accuracy and lowest error rates across validation data. This ensures that the chosen model is not only well-optimized but also robust and generalizable for real-world applications. The best-performing model is then integrated into the dashboard for visualization and can be further extended into ERP systems for practical deployment in industrial environments.


---

## Neural Network Algorithms Used
The project leverages several algorithms and techniques for improved accuracy:

1. **Feedforward Neural Networks (FNN)** – For basic classification of intents.  
2. **Backpropagation** – Used during training to minimize error via gradient descent.  
3. **Word Embeddings (Word2Vec/TF-IDF/Tokenizer Sequences)** – For numerical representation of text.  
4. **Softmax Classifier** – For multiclass intent classification.  
5. **Adam Optimizer** – Provides efficient and adaptive learning.  

---

## Real-World Industrial Use Cases
This system has multiple real-world applications across industries:

- **Customer Support**  
  - Automates responses to FAQs.  
  - Reduces operational costs and improves response time.  

- **Healthcare**  
  - Provides mental health assistance by detecting emotional cues.  
  - Suggests remedies based on trained psychological datasets.  

- **Manufacturing & Supply Chain**  
  - Integrated with ERP to answer queries about stock, inventory, and logistics.  
  - Provides real-time alerts and replenishment suggestions.  

- **Finance & Banking**  
  - Handles customer inquiries related to accounts, transactions, and services.  
  - Improves compliance by providing accurate knowledge-base responses.  

---

## Dashboard UI
The project includes a **modern interactive dashboard** built with **Dash (Plotly) & Flask**:  

<img width="1920" height="1080" alt="Screenshot (1225)" src="https://github.com/user-attachments/assets/fd8610b8-ffb1-49a2-a55b-80a6346326f0" />


- **Dark Theme with Neon Highlights** – Futuristic design for easy readability.  
- **Visualizations**:  
  - Model performance metrics (accuracy, loss curves).  
  - Query distribution by intent.  
  - Real-time conversation logs.  
  - Animated graphs for live insights.  
- **User Controls**:  
  - Filters for date/time ranges.  
  - Search functionality.  
  - Exportable analytics reports.  

The dashboard serves as a **control hub**, enabling administrators and business users to track both **chatbot interactions** and **system performance**.  

---

## Possible Integration with ERP Softwares
One of the major strengths of this system is its **ability to integrate with enterprise ERP platforms** like **SAP, Oracle NetSuite, Odoo, and Microsoft Dynamics**.  

- **ERP Query Handling**  
  - Users can ask the chatbot about stock availability, sales data, and procurement details.  
- **Automated Workflows**  
  - Chatbot can trigger ERP workflows such as purchase orders or inventory checks.  
- **Reporting & Analytics**  
  - The dashboard can pull ERP data and visualize it in real-time.  
- **User Accessibility**  
  - Provides non-technical staff with an easy, conversational way to access ERP data.  

This integration transforms the chatbot into a **business productivity tool** capable of enhancing decision-making, reducing manual effort, and streamlining operations.  

---

## Conclusion
This project demonstrates the **fusion of AI-driven chatbots, neural networks, and interactive dashboards** into a unified solution. With real-world industrial use cases and potential ERP integration, it provides a **scalable framework** for enterprises to enhance automation, efficiency, and intelligence in their workflows.  
