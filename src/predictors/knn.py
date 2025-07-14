import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Bibliotecas importadas com sucesso.")

# --- 1. Carregamento do Dataset ---
try:
    df = pd.read_csv('Obesity.csv')
    print("Dataset 'Obesity.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: O arquivo 'Obesity.csv' não foi encontrado.")
    exit()

# --- 2. Feature Engineering: Criação do IMC ---
df['IMC'] = df['Weight'] / (df['Height']**2)
print("Feature 'IMC' criada a partir de Peso e Altura.")

# --- 3. Separação de Features e Target ---
y = df['Obesity']
X = df.drop('Obesity', axis=1)

# --- 4. Identificação de Features Numéricas e Categóricas ---
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# --- 5. Divisão dos Dados em Conjuntos de Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dados divididos em {X_train.shape[0]} amostras de treino e {X_test.shape[0]} amostras de teste.")

# --- 6. Pipeline de Pré-processamento ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# --- 7. Criação e Treinamento do Pipeline do Modelo ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
])

print("\nIniciando o treinamento do modelo KNN...")
model_pipeline.fit(X_train, y_train)
print("Treinamento concluído.")

# --- 8. Avaliação do Modelo ---
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n-------------------------------------------")
print(f"Acurácia do Modelo: {accuracy * 100:.2f}%")
print("-------------------------------------------\n")
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("Relatório de classificação:\n", classification_report(y_test, y_pred))

# --- 9. Salvando o Modelo Treinado ---
joblib.dump(model_pipeline, 'KNN.pkl')
print("\nModelo salvo com sucesso como 'KNN.pkl'")
