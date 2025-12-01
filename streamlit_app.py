import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


# --------------------------
# Helpers
# --------------------------
def load_olist_sample(limit=20000):
    base = 'https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/'

    orders = pd.read_csv(
        base + 'olist_orders_dataset.csv',
        parse_dates=['order_purchase_timestamp','order_estimated_delivery_date','order_delivered_customer_date'],
        low_memory=False
    )
    items = pd.read_csv(base + 'olist_order_items_dataset.csv', low_memory=False)
    customers = pd.read_csv(base + 'ol
ist_customers_dataset.csv', low_memory=False)

    df = orders.merge(
        items.groupby('order_id').agg(
            n_items=('order_id','size'),
            n_unique_products=('product_id','nunique'),
            order_total_price=('price','sum')
        ).reset_index(),
        on='order_id',
        how='left'
    )

    df = df.merge(customers[['customer_id','customer_state']], on='customer_id', how='left')

    df['delayed'] = (
        (df['order_delivered_customer_date'] > df['order_estimated_delivery_date']) |
        df['order_delivered_customer_date'].isna()
    ).astype(int)

    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_to_estimate_days'] = (
        df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
    ).dt.days.fillna(0)

    df = df.dropna(subset=[
        'order_total_price','n_items','customer_state',
        'order_purchase_timestamp','order_estimated_delivery_date'
    ])

    return df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)


def prepare_X_y(df):
    features = [
        'order_total_price','n_items','n_unique_products',
        'purchase_to_estimate_days','purchase_hour','purchase_dayofweek',
        'customer_state'
    ]

    if 'n_unique_products' not in df.columns:
        df['n_unique_products'] = df.groupby('order_id')['product_id'].transform('nunique')

    X = df[features].copy()
    y = df['delayed'].astype(int).copy()

    X.fillna({
        'order_total_price': X['order_total_price'].median(),
        'n_items': 1,
        'n_unique_products': 1,
        'purchase_to_estimate_days': X['purchase_to_estimate_days'].median(),
        'purchase_hour': 0,
        'purchase_dayofweek': 0
    }, inplace=True)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_matrix = ohe.fit_transform(X[['customer_state']])
    states = ohe.categories_[0]

    # nomes das colunas serão só as siglas (SP, RJ, BA...)
    cat_cols = list(states)

    X_num = X.drop(columns=['customer_state']).reset_index(drop=True)
    X_processed = pd.concat(
        [X_num, pd.DataFrame(cat_matrix, columns=cat_cols)],
        axis=1
    )

    return X_processed, y, ohe, cat_cols


def train_demo_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    return clf, X_test, y_test, y_pred, y_proba


def predict_from_inputs(clf, ohe, cat_cols, inputs):
    df = pd.DataFrame([inputs])
    cat_matrix = ohe.transform(df[['customer_state']])
    df_num = df.drop(columns=['customer_state']).reset_index(drop=True)
    df_proc = pd.concat(
        [df_num, pd.DataFrame(cat_matrix, columns=cat_cols)],
        axis=1
    )

    proba = clf.predict_proba(df_proc)[:,1][0]
    pred = int(proba >= 0.5)
    return pred, proba


# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title='Modelo Olist - Previsão de Atraso', layout='centered')
st.title('📦 Previsão de Atraso em Entregas – Modelo Olist')

st.markdown("""
Este aplicativo demonstra todo o fluxo de uma solução de Data Science:

1. **Carregamento & Exploração dos Dados (EDA)**  
2. **Treinamento de Modelo (Random Forest)**  
3. **Visualizações importantes**  
4. **Interatividade: teste o modelo com exemplos reais ou valores manuais**
""")


mode = st.sidebar.selectbox(
    "Escolha o modo:",
    ["Demo (baixar e treinar)", "Upload CSV (pré-processado)", "Entrada manual"]
)


# =========================================================
# DEMO MODE
# =========================================================
if mode == "Demo (baixar e treinar)":
    st.info("No modo *Demo*, os dados públicos do Olist serão baixados automaticamente.")

    if st.button("Baixar dados e treinar modelo"):
        with st.spinner("Baixando dados e treinando modelo..."):
            df = load_olist_sample(limit=5000)
            st.success(f"Dados carregados: {len(df)} linhas.")

            # ----------- VISUALIZAÇÕES INTERATIVAS -----------
            st.subheader("📊 Análise Exploratória (EDA)")

            st.markdown("### Distribuição de pedidos atrasados")
            st.write("Este gráfico mostra quantos pedidos chegaram no prazo e quantos atrasaram.")
            fig1 = px.bar(df['delayed'].value_counts(), title="Atrasado (1) vs Não atrasado (0)")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### Distribuição por estado")
            st.write("Mostra de quais estados vêm a maior parte dos pedidos.")
            fig2 = px.bar(df['customer_state'].value_counts(), title="Pedidos por estado")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Preço total do pedido x atraso")
            st.write("Permite investigar se pedidos mais caros têm maior probabilidade de atrasar.")
            fig3 = px.scatter(
                df,
                x="order_total_price",
                y="delayed",
                opacity=0.5,
                title="Preço x Atraso"
            )
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### Número de itens x atraso")
            st.write("Ajuda a entender se pedidos com mais itens atrasam mais.")
            fig4 = px.scatter(
                df,
                x="n_items",
                y="delayed",
                opacity=0.5,
                title="Itens x Atraso"
            )
            st.plotly_chart(fig4, use_container_width=True)

            # ----------- Treinamento ----------
            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, X_test, y_test, y_pred, y_proba = train_demo_model(X, y)

            st.subheader("📈 Métricas do Modelo")
            st.write("F1-Score:", f1_score(y_test, y_pred))
            st.write("AUC:", roc_auc_score(y_test, y_proba))
            st.text(classification_report(y_test, y_pred, zero_division=0))

            # Feature importance com nomes das features limpos
            try:
                fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
                st.subheader("🌟 Importância das Features")
                st.write("Este gráfico mostra quais variáveis mais influenciam o modelo.")

                figFI = px.bar(fi, title="Importância das Features")
                st.plotly_chart(figFI, use_container_width=True)

            except:
                pass

            # armazenar modelo na sessão
            st.session_state['demo_clf'] = clf
            st.session_state['demo_ohe'] = ohe
            st.session_state['demo_cat_cols'] = cat_cols
            st.session_state['demo_sample'] = df

    # ---------- Predição usando pedido real ----------
    if st.session_state.get("demo_clf") is not None:
        df_sample = st.session_state['demo_sample']

        st.subheader("🧪 Testar modelo com um pedido real")
        pedido_id = st.selectbox(
            "Escolha um pedido real do dataset:",
            df_sample['order_id'].astype(str).tolist()
        )

        if st.button("Fazer previsão"):
            row = df_sample[df_sample['order_id'] == pedido_id].iloc[0]

            inputs = {
                'order_total_price': float(row['order_total_price']),
                'n_items': int(row['n_items']),
                'n_unique_products': int(row['n_unique_products']),
                'purchase_to_estimate_days': int(row['purchase_to_estimate_days']),
                'purchase_hour': int(row['purchase_hour']),
                'purchase_dayofweek': int(row['purchase_dayofweek']),
                'customer_state': str(row['customer_state'])
            }

            pred, proba = predict_from_inputs(
                st.session_state['demo_clf'],
                st.session_state['demo_ohe'],
                st.session_state['demo_cat_cols'],
                inputs
            )

            st.write(f"### Probabilidade de atraso: **{proba*100:.1f}%**")

            if pred == 1:
                st.error("🔴 **Previsão: o pedido provavelmente irá atrasar.**")
            else:
                st.success("🟢 **Previsão: o pedido provavelmente NÃO irá atrasar.**")


# =========================================================
# UPLOAD CSV MODE
# =========================================================
if mode == "Upload CSV (pré-processado)":
    st.info("Faça upload de um CSV já pré-processado.")

    uploaded = st.file_uploader("Selecione seu arquivo CSV", type=['csv'])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())

        if st.button("Treinar modelo"):
            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, X_test, y_test, y_pred, y_proba = train_demo_model(X, y)

            st.write("F1:", f1_score(y_test, y_pred))

            st.session_state['demo_clf'] = clf
            st.session_state['demo_ohe'] = ohe
            st.session_state['demo_cat_cols'] = cat_cols
            st.session_state['demo_sample'] = df


# =========================================================
# MANUAL INPUT MODE
# =========================================================
if mode == "Entrada manual":
    st.subheader("✏️ Previsão usando valores manuais")

    col1, col2 = st.columns(2)

    with col1:
        order_total_price = st.number_input('Preço total do pedido', min_value=0.0, value=50.0)
        n_items = st.number_input('Número de itens', min_value=1, value=1)
        n_unique_products = st.number_input('Produtos diferentes', min_value=1, value=1)
        purchase_to_estimate_days = st.number_input('Dias entre compra e estimativa', min_value=0, value=5)

    with col2:
        purchase_hour = st.number_input('Hora da compra', min_value=0, max_value=23, value=12)
        purchase_dayofweek = st.number_input('Dia da semana (0=Seg, 6=Dom)', min_value=0, max_value=6, value=2)
        customer_state = st.selectbox('Estado do cliente', [
            'SP','RJ','MG','BA','PR','RS','SC','CE','PE','PA','Other'
        ])

    if st.button("Prever"):
        if st.session_state.get("demo_clf") is None:
            st.warning("Treine o modelo primeiro (modo Demo ou Upload CSV).")
        else:
            inputs = {
                'order_total_price': float(order_total_price),
                'n_items': int(n_items),
                'n_unique_products': int(n_unique_products),
                'purchase_to_estimate_days': int(purchase_to_estimate_days),
                'purchase_hour': int(purchase_hour),
                'purchase_dayofweek': int(purchase_dayofweek),
                'customer_state': customer_state
            }

            pred, proba = predict_from_inputs(
                st.session_state['demo_clf'],
                st.session_state['demo_ohe'],
                st.session_state['demo_cat_cols'],
                inputs
            )

            st.write(f"### Probabilidade de atraso: **{proba*100:.1f}%**")

            if pred == 1:
                st.error("🔴 Pedido provavelmente ATRASARÁ")
            else:
                st.success("🟢 Pedido provavelmente NÃO atrasará")

st.markdown("---")
st.markdown("Grupo: José Matheus Simsen Lopes")
