# 🌍 Carbon Credits Analytics Platform

## 📊 **Análise Inteligente de Créditos de Carbono**

Plataforma de business intelligence para análise de **458,302 transações reais** de créditos de carbono, cobrindo **22.5 anos** de dados históricos (2002-2025) de registros globais.

---

## 🚀 **Features**

### **✅ Análises Disponíveis (100% Dados Reais)**
- **Análise Descritiva:** 69 categorias de projetos, 109 países, insights estatísticos
- **Testes Estatísticos:** Mann-Whitney U, Chi-square, análises de independência
- **Visualizações Interativas:** Distribuições, comparações, trends temporais
- **Market Intelligence:** Análises geográficas e por categoria

### **🔄 Em Desenvolvimento (Roadmap Q3 2025)**
- **Seasonal Activity Predictor:** Timing otimizado baseado em padrões históricos
- **Volatility Risk Calculator:** Avaliação de risco por categoria
- **Trend Analyzer:** Análise de crescimento/declínio de mercado
- **Volume Forecaster:** Previsões mensais com sazonalidade

---

## 📈 **Dados & Insights**

| Métrica | Valor |
|---------|-------|
| **Total Transações** | 458,302 |
| **Período Coberto** | 22.5 anos (2002-2025) |
| **Categorias de Projeto** | 69 diferentes |
| **Países Incluídos** | 109 |
| **Maior Categoria** | REDD+ (123,807 transações) |
| **Volume Total** | 2.8+ bilhões tCO₂ |

### **📊 Insights Descobertos:**
- **Sazonalidade:** Abril = pico de atividade, Agosto = baixa
- **Tendências:** Cookstove +73.5% crescimento, REDD+ -73.1% declínio
- **Volatilidade:** Rice Emission mais estável (CV=0.59), Wind mais volátil
- **Geografia:** India lidera em transações (99,294), Brasil em segundo (62,155)

---

## 💻 **Como Executar**

### **Pré-requisitos:**
- Python 3.8+
- pip

### **Instalação:**
```bash
# Clone o repositório
git clone [repository-url]
cd carbon-pricing

# Instale dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run src/app.py
```

### **Navegação:**
1. **📊 Comparative Overview** - Análises comparativas por categoria/país
2. **📈 Distribution Analysis** - Distribuições estatísticas e histogramas  
3. **🔬 Statistical Analysis** - Testes estatísticos avançados
4. **🧠 Market Analysis** - Business intelligence com dados reais

---

## 🏗️ **Estrutura do Projeto**

```
carbon-pricing/
├── data/                          # Dados CSV originais
├── src/
│   ├── app.py                     # Aplicação Streamlit principal
│   └── modules/
│       ├── analysis/              # Análises estatísticas reais
│       │   ├── descriptive.py     # Estatísticas descritivas
│       │   ├── inferential.py     # Testes estatísticos
│       │   └── modeling.py        # Framework ML
│       ├── predictive/            # Calculadoras preditivas (futuro)
│       ├── plotting/              # Visualizações
│       └── data_loader.py         # Carregamento de dados
├── docs/                          # Documentação e roadmaps
├── tests/                         # Testes unitários (futuro)
└── requirements.txt               # Dependências Python
```

---

## 📈 **Valor Comercial**

### **🎯 Target Market:**
- **Consultórias ESG** - Due diligence e risk assessment
- **Fundos de Investimento** - Portfolio optimization
- **Empresas de Trading** - Market timing e pricing strategy
- **Corporações** - Carbon offset strategy

### **💰 Modelo de Receita Projetado:**
- **Essential Analytics:** R$ 399/mês
- **Professional Suite:** R$ 1,299/mês  
- **Enterprise Solution:** R$ 4,999/mês

### **📊 Projeções Conservadoras:**
- **Ano 1:** R$ 300k - 1.4M receita
- **Break-even:** 12-18 meses
- **Market Size:** R$ 2-5B+ (mercado global ESG)

---

## 🛠️ **Stack Técnico**

- **Backend:** Python, Pandas, NumPy, SciPy
- **Frontend:** Streamlit, Plotly, Matplotlib
- **Data Science:** Scikit-learn, Statsmodels
- **Infrastructure:** CSV-based (migração futura para DB)

---

## 📋 **Próximos Passos**

### **Fase 1 (Semanas 1-2):** Quick Wins
- [ ] Seasonal Activity Predictor
- [ ] Volatility Risk Calculator

### **Fase 2 (Semanas 3-4):** Core Value  
- [ ] Trend Analyzer
- [ ] Volume Forecasting

### **Fase 3 (Semanas 5-6):** Advanced Features
- [ ] Geographic Expansion Predictor
- [ ] API Development

---

## 📜 **Disclaimer**

Todas as análises são baseadas em **dados históricos reais** de registros públicos de créditos de carbono. A plataforma não contém valores fictícios ou simulados. Resultados passados não garantem performance futura.

---

## 👥 **Contato**

Para mais informações sobre licenciamento empresarial ou partnerships:
- **Website:** [Em desenvolvimento]
- **Demo:** `streamlit run src/app.py`
- **Documentação:** Ver pasta `docs/`

---

**🌱 Transformando dados de carbono em inteligência acionável.** 