import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# -------------------------------------------------------------------
# Carregar dados Olist
# -------------------------------------------------------------------
def load_olist_sample(limit=20000):
    base = 'https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/'

    orders = pd.read_csv(base + 'olist_orders_dataset.csv',
                         parse_dates=['order_purchase_timestamp',
                                      'order_estimated_delivery_date',
                                      'order_delivered_customer_date'])

    items = pd.read_csv(base + 'olist_order_items_dataset.csv')
    customers = pd.read_csv(base + 'olist_customers_dataset.csv')

    df = orders.merge(
        items.groupby('order_id').agg(
            n_items=('order_id','size'),
            n_unique_products=('product_id', 'nunique'),
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

    df = df.dropna(subset=['order_total_price','n_items','customer_state'])

    return df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)

# -------------------------------------------------------------------
# Preparar dados
# -------------------------------------------------------------------
def prepare_X_y(df):
    features = [
        'order_total_price','n_items','n_unique_products',
        'purchase_to_estimate_days','purchase_hour','purchase_dayofweek',
        'customer_state'
    ]

    X = df[features].copy()
    y = df['delayed'].astype(int)

    X = X.fillna({
        'order_total_price': X['order_total_price'].median(),
        'n_items': 1,
        'n_unique_products': 1,
        'purchase_to_estimate_days': X['purchase_to_estimate_days'].median(),
        'purchase_hour': 0,
        'purchase_dayofweek': 0
    })

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat = ohe.fit_transform(X[['customer_state']])
    cat_cols = list(ohe.get_feature_names_out(['customer_state']))

    X_final = pd.concat(
        [X.drop(columns=['customer_state']).reset_index(drop=True),
         pd.DataFrame(cat, columns=cat_cols)],
        axis=1
    )

    return X_final, y, ohe, cat_cols

# -------------------------------------------------------------------
# Treinar modelo
# -------------------------------------------------------------------
def train_demo_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    return clf, X_test, y_test, y_pred, y_proba

# -------------------------------------------------------------------
# Prever
# -------------------------------------------------------------------
def predict_from_inputs(clf, ohe, cat_cols, inputs):
    df = pd.DataFrame([inputs])
    cat = ohe.transform(df[['customer_state']])
    df_num = df.drop(columns=['customer_state'])

    df_proc = pd.concat(
        [df_num.reset_index(drop=True), pd.DataFrame(cat, columns=cat_cols)], axis=1
    )

    proba = clf.predict_proba(df_proc)[:,1][0]
    pred = int(proba >= 0.5)
    return pred, proba

# -------------------------------------------------------------------
# INTERFACE STREAMLIT
# -------------------------------------------------------------------
st.set_page_config(page_title='Previsão de Atraso - Olist', layout='centered')
st.title('📦 Previsão de Entrega Atrasada - Olist')

st.markdown("""
Bem-vindo!  
Este app demonstra como transformar um notebook em uma interface **Streamlit**.

Você pode:
- 👨‍🏫 Treinar um modelo de demonstração com dados Olist  
- 📤 Enviar seu próprio CSV  
- ✏️ Inserir valores manualmente e obter previsões  
""")

# Sidebar
st.sidebar.title("⚙️ Opções")
mode = st.sidebar.radio(
    "Escolha o modo:",
    ["Demo (baixar e treinar)", "Upload CSV", "Entrada manual"]
)

# -------------------------------------------------------------------
# MODO DEMONSTRAÇÃO
# -------------------------------------------------------------------
if mode == "Demo (baixar e treinar)":
    st.subheader("🔄 Treinar modelo de demonstração")

    if st.button("Baixar dados e treinar modelo"):
        with st.spinner("Carregando dados e treinando modelo..."):
            df = load_olist_sample(limit=6000)

            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, X_test, y_test, y_pred, y_proba = train_demo_model(X, y)

            st.success("Modelo treinado com sucesso!")
            st.write("### 📊 Desempenho do modelo")
            st.write("**F1 Score:**", f1_score(y_test, y_pred))
            st.write("**AUC:**", roc_auc_score(y_test, y_proba))

            st.session_state.update({
                'demo_clf': clf,
                'demo_ohe': ohe,
                'demo_cat_cols': cat_cols,
                'demo_sample': df
            })

    if st.session_state.get('demo_clf'):
        st.divider()
        st.subheader("📌 Fazer previsão com um exemplo real")

        df_sample = st.session_state['demo_sample']
        idx = st.number_input("Escolha o índice da linha:", 0, len(df_sample)-1, 0)

        if st.button("Prever exemplo"):
            row = df_sample.iloc[int(idx)]

            inputs = {
                'order_total_price': float(row['order_total_price']),
                'n_items': int(row['n_items']),
                'n_unique_products': int(row['n_unique_products']),
                'purchase_to_estimate_days': int(row['purchase_to_estimate_days']),
                'purchase_hour': int(row['purchase_hour']),
                'purchase_dayofweek': int(row['purchase_dayofweek']),
                'customer_state': row['customer_state']
            }

            pred, proba = predict_from_inputs(
                st.session_state['demo_clf'],
                st.session_state['demo_ohe'],
                st.session_state['demo_cat_cols'],
                inputs
            )

            st.write("### Resultado:")
            st.write("📦 **Atraso:**", "Sim" if pred == 1 else "Não")
            st.write(f"🔢 Probabilidade: **{proba:.2f}**")

# -------------------------------------------------------------------
# MODO CSV
# -------------------------------------------------------------------
if mode == "Upload CSV":
    st.subheader("📤 Enviar CSV pré-processado")

    uploaded = st.file_uploader("Envie um arquivo CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Prévia dos dados:", df.head())

        if st.button("Treinar modelo com CSV"):
            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, _, _, _, _ = train_demo_model(X, y)

            st.success("Modelo treinado!")
            st.session_state.update({
                'demo_clf': clf,
                'demo_ohe': ohe,
                'demo_cat_cols': cat_cols
            })

# -------------------------------------------------------------------
# MODO MANUAL
# -------------------------------------------------------------------
if mode == "Entrada manual":
    st.subheader("✏️ Entrada manual de dados")

    col1, col2 = st.columns(2)

    with col1:
        order_total_price = st.number_input("Valor total do pedido", min_value=0.0, value=100.0)
        n_items = st.number_input("Quantidade de itens", min_value=1, value=1)
        n_unique_products = st.number_input("Produtos únicos", min_value=1, value=1)

    with col2:
        purchase_to_estimate_days = st.number_input("Dias até a entrega prevista", min_value=0, value=5)
        purchase_hour = st.number_input("Hora da compra (0–23)", min_value=0, max_value=23, value=12)
        purchase_dayofweek = st.number_input("Dia da semana (0=Seg)", min_value=0, max_value=6, value=2)
        customer_state = st.text_input("Estado (UF)", "SP")

    if st.button("Prever atraso"):
        if not st.session_state.get('demo_clf'):
            st.warning("Treine o modelo primeiro no modo Demo ou CSV.")
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

            st.write("### Resultado:")
            st.write("📦 **Atraso:**", "Sim" if pred == 1 else "Não")
            st.write(f"🔢 Probabilidade: **{proba:.2f}**")

# Rodapé
st.divider()
st.caption("App criado para demonstração de modelo — Streamlit")
