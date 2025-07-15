# --- 1. Importação das Bibliotecas Necessárias ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Bibliotecas importadas com sucesso.")

# --- 2. Carregamento do Dataset ---
try:
    df = pd.read_csv('data/Obesity.csv')
    print("Dataset 'Obesity.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'Obesity.csv' não foi encontrado.")
    exit()

# --- 3. Feature Engineering: Criação do IMC ---
df['IMC'] = df['Weight'] / (df['Height']**2)
print("Feature 'IMC' criada a partir de Peso e Altura.")

# --- 4. Separação de Features e Target ---
# A variável alvo (o que queremos prever) é 'Obesity'
y = df['Obesity']
# As features (variáveis preditoras) são todas as outras colunas, incluindo o IMC
X = df.drop('Obesity', axis=1)

# --- 5. Identificação de Features Numéricas e Categóricas ---
# Identificação automática das colunas numéricas e categóricas
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# --- 6. Divisão dos Dados em Conjuntos de Treino e Teste ---
# Dividimos os dados para treinar o modelo e depois testá-lo em dados não vistos.
# 80% para treino, 20% para teste.
# 'stratify=y' garante que a proporção das classes de obesidade seja a mesma nos dois conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dados divididos em {X_train.shape[0]} amostras de treino e {X_test.shape[0]} amostras de teste.")

# --- 7. Pipeline de Pré-processamento ---
# Criação do transformador que aplica as etapas corretas para cada tipo de coluna
preprocessor = ColumnTransformer(
    transformers=[
        # Etapa 'num': Aplica padronização (StandardScaler) às features numéricas.
        # Isso coloca todas as variáveis numéricas na mesma escala.
        ('num', StandardScaler(), numerical_features),
        
        # Etapa 'cat': Aplica codificação One-Hot (OneHotEncoder) às features categóricas.
        # Isso transforma categorias (ex: 'Male', 'Female') em formato numérico.
        # 'handle_unknown='ignore'' previne erros se uma categoria rara aparecer só no teste.
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# --- 8. Criação e Treinamento do Pipeline do Modelo ---
# O pipeline completo une o pré-processador e o modelo de classificação.
# Isso garante que os dados de teste sejam transformados da mesma forma que os de treino.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Treinamento do modelo com os dados de treino
print("\nIniciando o treinamento do modelo RandomForest...")
model_pipeline.fit(X_train, y_train)
print("Treinamento concluído.")

# --- 9. Avaliação do Modelo ---
# Realiza previsões nos dados de teste para avaliar a performance
y_pred = model_pipeline.predict(X_test)

# Calcula e exibe a acurácia geral
accuracy = accuracy_score(y_test, y_pred)
print("\n-------------------------------------------")
print(f"Acurácia do Modelo: {accuracy * 100:.2f}%")
print("-------------------------------------------\n")

# Exibe o relatório de classificação detalhado com precisão, recall e f1-score
print("Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))

print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))


# --- 10. Salvando o Modelo Treinado para Deploy ---
# Salva o pipeline completo em um único arquivo.
# Este arquivo será carregado pela aplicação Streamlit para fazer novas previsões.
joblib.dump(model_pipeline, 'RandomForest.pkl')
print("\nModelo salvo com sucesso como 'RandomForest.pkl'")