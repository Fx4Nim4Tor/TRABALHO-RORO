import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import plotly.express as px

st.set_page_config(page_title="Modelo de Atraso de Entregas", layout="wide")

st.title("📦 Predição de Atrasos em Entregas – Olist")
st.write("""
Este aplicativo permite:
- Explorar dados reais da Olist  
- Visualizar gráficos interativos  
- Treinar um modelo de Machine Learning  
- Testar o modelo com dados reais ou valores manuais  
""")

# --------------------------------------------------------------------
# 1) Carregar dados
# --------------------------------------------------------------------
@st.cache_data
def load_demo_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/BrunoS3D/olist-mini-dataset/main/olist_orders_dataset_mini.csv"
    )
    df['delayed'] = (df['order_delivered_customer_date'] >
                     df['order_estimated_delivery_date']).astype(int)
    df['purchase_hour'] = pd.to_datetime(df['order_purchase_timestamp']).dt.hour
    df['purchase_dayofweek'] = pd.to_datetime(df['order_purchase_timestamp']).dt.dayofweek
    return df

df = load_demo_data()
st.success("Dados carregados com sucesso!")

# --------------------------------------------------------------------
# 2) Visualizações (EDA)
# --------------------------------------------------------------------
st.header("📊 Exploração dos Dados (EDA)")

# --- Distribuição de atrasos ---
st.markdown("### 🔴 Distribuição de Atrasos")
st.write("Mostra quantos pedidos atrasaram e quantos foram entregues dentro do prazo.")
fig_delay = px.bar(
    df['delayed'].value_counts().reset_index(),
    x="index",
    y="delayed",
    labels={"index": "Atrasou?", "delayed": "Quantidade"},
    title="Distribuição de pedidos atrasados",
)
st.plotly_chart(fig_delay, use_container_width=True)

# --- Distribuição por estado ---
st.markdown("### 🗺️ Distribuição de Pedidos por Estado")
st.write("Ajuda a visualizar em quais estados há mais compras registradas.")
fig_state = px.bar(
    df['customer_state'].value_counts().reset_index(),
    x="index",
    y="customer_state",
    labels={"index": "Estado", "customer_state": "Quantidade"},
    title="Distribuição por estado",
)
st.plotly_chart(fig_state, use_container_width=True)

# --- Preço total x atraso ---
st.markdown("### 💰 Preço total do pedido x Atraso")
st.write("Mostra se pedidos mais caros têm mais probabilidade de atrasar.")
fig_price = px.scatter(
    df,
    x="order_total_price",
    y="delayed",
    color="delayed",
    labels={"delayed": "Atrasou?", "order_total_price": "Preço total"},
    title="Preço total x atraso",
)
st.plotly_chart(fig_price, use_container_width=True)

# --- Número de itens x atraso ---
st.markdown("### 📦 Quantidade de itens x Atraso")
st.write("Verifica se pedidos com mais itens tendem a atrasar mais.")
fig_items = px.scatter(
    df,
    x="n_items",
    y="delayed",
    color="delayed",
    labels={"delayed": "Atrasou?", "n_items": "Número de itens"},
    title="Número de itens x atraso",
)
st.plotly_chart(fig_items, use_container_width=True)

# --------------------------------------------------------------------
# 3) Preparar dados para treino do modelo
# --------------------------------------------------------------------
def prepare_X_y(df):
    X = df[[
        "order_total_price",
        "n_items",
        "n_unique_products",
        "purchase_to_estimate_days",
        "purchase_hour",
        "purchase_dayofweek",
        "customer_state"
    ]]
    y = df["delayed"]

    cat_cols = ["customer_state"]
    num_cols = X.drop(columns=cat_cols).columns

    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_ohe = ohe.fit_transform(X[cat_cols])

    X_final = np.concatenate([X[num_cols], X_ohe], axis=1)

    return X_final, y, ohe, cat_cols, num_cols


X, y, ohe, cat_cols, num_cols = prepare_X_y(df)

# --------------------------------------------------------------------
# 4) Treinar modelo
# --------------------------------------------------------------------
st.header("🧠 Treinamento do Modelo")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
proba_test = clf.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, preds)
auc = roc_auc_score(y_test, proba_test)

st.write(f"**F1-Score:** `{f1:.3f}`")
st.write(f"**AUC:** `{auc:.3f}`")

# --------------------------------------------------------------------
# 5) Importância das Features
# --------------------------------------------------------------------
st.subheader("📌 Importância das Features")

# nomes da one-hot
ohe_features = ohe.get_feature_names_out(cat_cols)

# limpar nomes: customer_state_RS → RS
ohe_clean = [f.split("_")[-1] for f in ohe_features]

feature_names = list(num_cols) + ohe_clean
importances = clf.feature_importances_

df_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importância": importances
}).sort_values("Importância", ascending=False)

fig_imp = px.bar(
    df_imp,
    x="Importância",
    y="Feature",
    orientation="h",
    title="Importância das Features (com nomes limpos)"
)
st.plotly_chart(fig_imp, use_container_width=True)

# --------------------------------------------------------------------
# 6) Predição interativa
# --------------------------------------------------------------------
st.header("🧪 Testar o Modelo Interativamente")

tab1, tab2 = st.tabs(["🔎 Usar um pedido real", "✍️ Inserir valores manualmente"])

# ---------------- TAB 1: Escolher linha real ----------------------
with tab1:
    st.subheader("Escolha um pedido real para prever")

    pedido_ids = df["order_id"].tolist()
    escolha = st.selectbox("Selecione o pedido:", pedido_ids)

    row = df[df["order_id"] == escolha].iloc[0]

    if st.button("📌 Prever atraso para este pedido"):
        inputs = {
            "order_total_price": row["order_total_price"],
            "n_items": row["n_items"],
            "n_unique_products": row["n_unique_products"],
            "purchase_to_estimate_days": row["purchase_to_estimate_days"],
            "purchase_hour": row["purchase_hour"],
            "purchase_dayofweek": row["purchase_dayofweek"],
            "customer_state": row["customer_state"]
        }

        # preparar entrada manual
        X_manual = np.array([[inputs["order_total_price"],
                              inputs["n_items"],
                              inputs["n_unique_products"],
                              inputs["purchase_to_estimate_days"],
                              inputs["purchase_hour"],
                              inputs["purchase_dayofweek"]]])

        # one-hot
        ohe_input = ohe.transform([[inputs["customer_state"]]])
        X_final_pred = np.concatenate([X_manual, ohe_input], axis=1)

        pred = clf.predict(X_final_pred)[0]
        proba = clf.predict_proba(X_final_pred)[0][1]

        st.write(f"### Probabilidade de atraso: **{proba*100:.1f}%**")

        if pred == 1:
            st.error("🔴 O modelo prevê **atraso** para este pedido.")
        else:
            st.success("🟢 O modelo prevê **entrega dentro do prazo**.")

# ---------------- TAB 2: Inserir valores manualmente ----------------------
with tab2:
    st.subheader("Insira os valores para prever")

    col1, col2, col3 = st.columns(3)

    order_total_price = col1.number_input("Preço total (R$)", 0.0, 5000.0, 200.0)
    n_items = col2.number_input("Número de itens", 1, 20, 3)
    n_unique_products = col3.number_input("Produtos únicos", 1, 20, 2)

    col4, col5, col6 = st.columns(3)

    purchase_to_estimate_days = col4.number_input("Dias até entrega estimada", 1, 50, 10)
    purchase_hour = col5.number_input("Hora da compra", 0, 23, 14)
    purchase_dayofweek = col6.selectbox("Dia da semana", list(range(7)))

    customer_state = st.selectbox("Estado do cliente", sorted(df["customer_state"].unique()))

    if st.button("📌 Prever com valores inseridos"):
        X_manual = np.array([[order_total_price, n_items, n_unique_products,
                              purchase_to_estimate_days, purchase_hour,
                              purchase_dayofweek]])

        ohe_input = ohe.transform([[customer_state]])
        X_final_pred = np.concatenate([X_manual, ohe_input], axis=1)

        pred = clf.predict(X_final_pred)[0]
        proba = clf.predict_proba(X_final_pred)[0][1]

        st.write(f"### Probabilidade de atraso: **{proba*100:.1f}%**")

        if pred == 1:
            st.error("🔴 O modelo prevê **atraso** para este pedido.")
        else:
            st.success("🟢 O modelo prevê **entrega dentro do prazo**.")
