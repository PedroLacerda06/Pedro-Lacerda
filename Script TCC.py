import pandas as pd
import numpy as np
import random

# Função para gerar nomes genéricos
def gerar_nome_generico(n):
    primeiro_nome = ['João', 'Maria', 'Pedro', 'Ana', 'Lucas', 'Julia', 'Carlos', 'Beatriz', 'Rafael', 'Fernanda']
    sobrenome = ['Silva', 'Souza', 'Oliveira', 'Pereira', 'Costa', 'Almeida', 'Nogueira', 'Barbosa', 'Cardoso', 'Santos']
    return [f"{random.choice(primeiro_nome)} {random.choice(sobrenome)}" for _ in range(n)]

# Lista de estados brasileiros
estados = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']

# Geração da base de dados
np.random.seed(42)  # Para reprodutibilidade
n = 1000

dados = {
    'Nome': gerar_nome_generico(n),
    'Estado': np.random.choice(estados, n),
    'Ano da primeira dívida': np.random.randint(2000, 2024, n),
    'Valor da dívida (R$)': np.round(np.random.uniform(100, 50000, n), 2),
    'Idade': np.random.randint(18, 81, n),
    'Ensino Superior': np.random.choice(['Sim', 'Não'], n, p=[0.3, 0.7])  # 30% das pessoas têm ensino superior
}

df = pd.DataFrame(dados)

# Introduzindo intencionalmente alguns valores ausentes e duplicados para tratamento posterior
df.loc[np.random.choice(df.index, 10), 'Valor da dívida (R$)'] = np.nan  # Alguns valores ausentes na dívida
df = pd.concat([df, df.sample(5)])  # Adicionando duplicatas intencionais

# Quantidade total de registros
total_registros = len(df)
print(f"Total de registros: {total_registros}")

# Verificando duplicatas
duplicatas = df[df.duplicated()]
print(f"Quantidade de registros duplicados: {len(duplicatas)}")

# Removendo duplicatas
df_sem_duplicatas = df.drop_duplicates()

# Verificando valores ausentes
valores_ausentes = df_sem_duplicatas.isnull().sum()
print(f"Valores ausentes por coluna:\n{valores_ausentes}")

# Preenchendo valores ausentes com a mediana da dívida
mediana_divida = df_sem_duplicatas['Valor da dívida (R$)'].median()
df_sem_duplicatas['Valor da dívida (R$)'].fillna(mediana_divida, inplace=True)

# Verificando quantos registros restaram após o tratamento
registros_finais = len(df_sem_duplicatas)
print(f"Registros finais após tratamento: {registros_finais}")


from scipy import stats

# Separando os dados de acordo com a escolaridade
divida_superior = df_sem_duplicatas[df_sem_duplicatas['Ensino Superior'] == 'Sim']['Valor da dívida (R$)']
divida_sem_superior = df_sem_duplicatas[df_sem_duplicatas['Ensino Superior'] == 'Não']['Valor da dívida (R$)']

# Teste t de Student
t_stat, p_value = stats.ttest_ind(divida_superior, divida_sem_superior, equal_var=False)

print(f"Estatística t: {t_stat}")
print(f"p-valor: {p_value}")

 #Decisão: Se o p-valor for menor que 0.05, rejeitamos a hipótese nula
if p_value < 0.05:
    print("Rejeitamos a hipótese nula. A média das dívidas é significativamente diferente.")
else:
  print("Não rejeitamos a hipótese nula. A média das dívidas não é significativamente diferente.")
    
# Intervalo de confiança para a média do valor da dívida
mean_divida = df_sem_duplicatas['Valor da dívida (R$)'].mean()
std_divida = df_sem_duplicatas['Valor da dívida (R$)'].std()
n_divida = len(df_sem_duplicatas['Valor da dívida (R$)'])

# Intervalo de confiança de 95%
conf_interval = stats.t.interval(0.95, df=n_divida-1, loc=mean_divida, scale=std_divida/np.sqrt(n_divida))

print(f"Intervalo de confiança de 95% para a média do valor da dívida: {conf_interval}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# Separando as variáveis explicativas (X) e a variável alvo (y)
X = df_sem_duplicatas[['Idade', 'Valor da dívida (R$)', 'Ano da primeira dívida']]
X = pd.get_dummies(df_sem_duplicatas[['Estado']], drop_first=True).join(X)  # Variáveis categóricas (Estado) transformadas em dummies
y = df_sem_duplicatas['Ensino Superior'].map({'Sim': 1, 'Não': 0})  # Variável alvo binária

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalando os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Teste com os três algoritmos

# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
auc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])

# 2. Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# 3. Support Vector Machine (SVM)
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
auc_svm = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])

# Comparando as acurácias e AUCs dos três modelos
{
    "Logistic Regression": {"Accuracy": accuracy_log_reg, "AUC": auc_log_reg},
    "Random Forest": {"Accuracy": accuracy_rf, "AUC": auc_rf},
    "SVM": {"Accuracy": accuracy_svm, "AUC": auc_svm}
}
