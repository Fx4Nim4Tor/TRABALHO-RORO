
# Streamlit app - Demo para notebook Olist

Este repositório contém um exemplo de aplicação Streamlit que demonstra como transformar seu notebook (PROJETO_RORO.ipynb) em um app interativo.

## O que o app faz
- Baixa uma amostra pública do dataset Olist do GitHub (modo *Demo*), gera feições simples e treina um modelo leve (RandomForest) para previsão de atraso de pedidos.
- Permite upload de CSV pré-processado para treinar um modelo rápido no navegador/servidor.
- Permite entrada manual de features para ver a previsão do modelo treinado na sessão.
- Contém instruções de uso.

## Como executar
1. Instale dependências:
```bash
pip install -r requirements.txt
```
2. Execute o app:
```bash
streamlit run streamlit_app.py
```

## Nota importante
- Este é um **demo**. Para usar o seu modelo treinado no notebook, salve o pipeline com `joblib.dump(best_model, 'best_model.pkl')` e modifique o app para carregar o pipeline com `joblib.load('best_model.pkl')`. Assim o app usará exatamente seu pré-processamento e classificador treinado.
