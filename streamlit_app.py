import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# --- Helpers ---------------------------------------------------------------
def load_olist_sample(limit=20000):
    """
    Download a few Olist tables from GitHub and prepare a small dataset for demo.
    This function will be executed when the user runs the Streamlit app locally (internet required).
    """
    base = 'https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/'
    orders = pd.read_csv(base + 'olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp','order_estimated_delivery_date','order_delivered_customer_date'], low_memory=False)
    items = pd.read_csv(base + 'olist_order_items_dataset.csv', low_memory=False)
    customers = pd.read_csv(base + 'olist_customers_dataset.csv', low_memory=False)

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

    df = df.dropna(subset=['order_total_price','n_items','customer_state','order_purchase_timestamp','order_estimated_delivery_date'])

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

    X['order_total_price'] = X['order_total_price'].fillna(X['order_total_price'].median())
    X['n_items'] = X['n_items'].fillna(1)
    X['n_unique_products'] = X['n_unique_products'].fillna(1)
    X['purchase_to_estimate_days'] = X['purchase_to_estimate_days'].fillna(X['purchase_to_estimate_days'].median())
    X['purchase_hour'] = X['purchase_hour'].fillna(0)
    X['purchase_dayofweek'] = X['purchase_dayofweek'].fillna(0)

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat = ohe.fit_transform(X[['customer_state']])
    cat_cols = list(ohe.get_feature_names_out(['customer_state']))

    X_num = X.drop(columns=['customer_state']).reset_index(drop=True)
    X_processed = pd.concat([X_num, pd.DataFrame(cat, columns=cat_cols)], axis=1)

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
    cat = ohe.transform(df[['customer_state']])
    df_num = df.drop(columns=['customer_state'])
    df_proc = pd.concat(
        [df_num.reset_index(drop=True), pd.DataFrame(cat, columns=cat_cols)],
        axis=1
    )

    proba = clf.predict_proba(df_proc)[:,1][0]
    pred = int(proba >= 0.5)

    return pred, proba

# --- Streamlit UI ---------------------------------------------------------
st.set_page_config(page_title='Demo Olist - Streamlit', layout='centered')
st.title('Demo: App Streamlit para modelo Olist')

st.markdown("""
Este app demonstra como transformar o seu notebook em uma aplicação **Streamlit**.

Funcionalidades:
- Baixar subconjunto do dataset Olist para demonstração
- Treinar modelo simples rápido
- Entrada manual de dados
- Previsão baseada em amostra real
- Métricas: F1, AUC, relatório de classificação
""")

st.sidebar.header('Modo de uso')
mode = st.sidebar.selectbox(
    'Escolha o modo:',
    ['Demo (baixar e treinar)', 'Upload CSV (pré-processado)', 'Entrada manual']
)

# ---------------- DEMO MODE -----------------
if mode == 'Demo (baixar e treinar)':
    st.info('No modo Demo o app baixa dados públicos do Olist e treina um modelo leve.')

    if st.button('Baixar dados e treinar modelo'):
        with st.spinner('Baixando e treinando...'):
            df = load_olist_sample(limit=8000)
            st.success(f'Dados carregados: {len(df)} linhas.')

            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, X_test, y_test, y_pred, y_proba = train_demo_model(X, y)

            st.subheader('Métricas do Modelo')
            st.write('F1:', f1_score(y_test, y_pred))
            st.write('AUC:', roc_auc_score(y_test, y_proba))
            st.write(classification_report(y_test, y_pred, zero_division=0))

            try:
                fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
                st.subheader('Importância das Features')
                st.table(fi.reset_index().rename(columns={'index':'feature',0:'importance'}))
            except:
                pass

            st.session_state['demo_clf'] = clf
            st.session_state['demo_ohe'] = ohe
            st.session_state['demo_cat_cols'] = cat_cols
            st.session_state['demo_sample'] = df

    if st.session_state.get('demo_clf') is not None:
        st.success('Modelo de demonstração carregado.')
        df_sample = st.session_state['demo_sample']

        st.subheader('Prever usando uma linha de exemplo')
        idx = st.number_input(
            'Índice (0 até N)',
            min_value=0,
            max_value=len(df_sample)-1,
            value=0
        )

        if st.button('Prever exemplo'):
            row = df_sample.iloc[int(idx)]
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

            st.write('Predição (1 = atraso):', pred)
            st.write(f'Probabilidade: {proba:.3f}')

# ---------------- UPLOAD MODE -----------------
if mode == 'Upload CSV (pré-processado)':
    st.info('Faça upload de um CSV com as colunas corretas.')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write('Preview do CSV:')
        st.dataframe(df.head())

        if st.button('Treinar modelo no CSV'):
            X, y, ohe, cat_cols = prepare_X_y(df)
            clf, X_test, y_test, y_pred, y_proba = train_demo_model(X, y)

            st.write('F1:', f1_score(y_test, y_pred))

            st.session_state['demo_clf'] = clf
            st.session_state['demo_ohe'] = ohe
            st.session_state['demo_cat_cols'] = cat_cols
            st.session_state['demo_sample'] = df

# ---------------- MANUAL MODE -----------------
if mode == 'Entrada manual':
    st.markdown('Insira os valores para previsão:')

    col1, col2 = st.columns(2)
    with col1:
        order_total_price = st.number_input('order_total_price', min_value=0.0, value=50.0)
        n_items = st.number_input('n_items', min_value=1, value=1)
        n_unique_products = st.number_input('n_unique_products', min_value=1, value=1)
        purchase_to_estimate_days = st.number_input('purchase_to_estimate_days', min_value=0, value=5)

    with col2:
        purchase_hour = st.number_input('purchase_hour', min_value=0, max_value=23, value=12)
        purchase_dayofweek = st.number_input('purchase_dayofweek', min_value=0, max_value=6, value=2)
        customer_state = st.selectbox('customer_state', ['SP','RJ','MG','BA','PR','RS','SC','CE','PE','PA','Other'])

    if st.button('Prever'):
        if st.session_state.get('demo_clf') is None:
            st.warning('Treine o modelo primeiro (modo Demo ou Upload CSV).')
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

            st.write('Predição (1 = atraso):', pred)
            st.write(f'Probabilidade: {proba:.3f}')

st.markdown('---')
st.markdown('**Como usar:** Instale as dependências e rode `streamlit run streamlit_app.py`.')
