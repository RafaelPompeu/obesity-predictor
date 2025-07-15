# --- 1. Importação das Bibliotecas ---
import streamlit as st
import pandas as pd
import joblib
import base64

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA (deve ser o PRIMEIRO comando Streamlit após os imports)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CARREGAMENTO DO MODELO (mantemos UMA única função cacheada)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    """Carrega o pipeline de machine learning salvo."""
    return joblib.load(model_path)

# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------

def get_image_as_base64(file_path: str) -> str:
    """Retorna representação Base64 de uma imagem local."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def traduzir_respostas_para_ingles(input_data: dict) -> dict:
    """Traduz respostas do formulário PT‑BR → EN (compatível com o modelo)."""
    traducoes = {
        "Gender": {"Masculino": "Male", "Feminino": "Female", "Male": "Male", "Female": "Female"},
        "family_history": {"Sim": "yes", "Não": "no", "yes": "yes", "no": "no"},
        "family_history_with_overweight": {"Sim": "yes", "Não": "no", "yes": "yes", "no": "no"},
        "FAVC": {"Sim": "yes", "Não": "no", "yes": "yes", "no": "no"},
        "CAEC": {
            "Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always",
            "no": "no", "Sometimes": "Sometimes", "Frequently": "Frequently", "Always": "Always",
        },
        "SMOKE": {"Sim": "yes", "Não": "no", "yes": "yes", "no": "no"},
        "SCC": {"Sim": "yes", "Não": "no", "yes": "yes", "no": "no"},
        "CALC": {
            "Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always",
            "no": "no", "Sometimes": "Sometimes", "Frequently": "Frequently", "Always": "Always",
        },
        "MTRANS": {
            "Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhada": "Walking",
            "Motocicleta": "Motorbike", "Bicicleta": "Bike",
            "Public_Transportation": "Public_Transportation", "Automobile": "Automobile",
            "Walking": "Walking", "Motorbike": "Motorbike", "Bike": "Bike",
        },
    }

    for campo, valor in input_data.items():
        if campo in traducoes and valor in traducoes[campo]:
            input_data[campo] = traducoes[campo][valor]
    return input_data


def traduzir_predicao_para_portugues(prediction_text: str) -> str:
    """Traduz saída do modelo EN → PT‑BR para exibição."""
    traducoes = {
        "Insufficient Weight": "Peso Insuficiente",
        "Normal Weight": "Peso Normal",
        "Overweight Level I": "Sobrepeso Nível I",
        "Overweight Level II": "Sobrepeso Nível II",
        "Obesity Type I": "Obesidade Tipo I",
        "Obesity Type II": "Obesidade Tipo II",
        "Obesity Type III": "Obesidade Tipo III",
    }
    return traducoes.get(prediction_text, prediction_text)

# -----------------------------------------------------------------------------
# CABEÇALHO DA APLICAÇÃO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='background-color:#0d6efd;padding:16px;border-radius:8px;margin-bottom:16px'>
        <h2 style='color:white;text-align:center;margin:0;'>Bem-vindo ao Sistema Preditivo de Obesidade</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# INTERFACE EM ABAS
# -----------------------------------------------------------------------------
abas = st.tabs(["Predição", "Dashboard Power BI"])

# =====================================================================
# ABA 1 – PREDIÇÃO
# =====================================================================
with abas[0]:
    # ------------------------------------------------------------------
    # Escolha e carregamento do modelo
    # ------------------------------------------------------------------
    model_options = {
        "KNN": "../models/KNN.pkl",
        "Random Forest": "../models/RandomForest.pkl",
        "SVM": "../models/SVM.pkl",
    }

    st.sidebar.header("Configuração do Modelo")
    selected_model_name = st.sidebar.selectbox(
        "Selecione o modelo de Machine Learning", list(model_options.keys()), index=1
    )
    selected_model_path = model_options[selected_model_name]

    try:
        model = load_model(selected_model_path)
    except FileNotFoundError:
        st.error(
            f"Arquivo do modelo não encontrado: '{selected_model_path}'.\n\nVerifique se o arquivo existe no diretório correto."
        )
        st.stop()

    # ------------------------------------------------------------------
    # Formulário de entrada de dados
    # ------------------------------------------------------------------
    st.title("🩺 Sistema Preditivo para Níveis de Obesidade")
    st.markdown(
        """
        Esta ferramenta foi desenvolvida para apoiar a equipe médica no diagnóstico de níveis de obesidade.\
        Utilizando um modelo de Machine Learning, o sistema analisa as informações do paciente para fornecer uma predição.\
        **Instruções:** Preencha os campos abaixo com os dados do paciente e clique em *Prever* para obter o resultado.
        """
    )
    st.info(
        "**Aviso:** Esta é uma ferramenta de apoio à decisão e não substitui o diagnóstico clínico realizado por um profissional de saúde qualificado."
    )

    with st.container():
        st.header("Formulário de Dados do Paciente")
        col1, col2, col3 = st.columns(3)

        # Informações Pessoais – col1
        with col1:
            age = st.number_input("Idade", min_value=1, max_value=100, value=25)
            gender = st.selectbox("Gênero", ["Masculino", "Feminino"])
            height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, format="%.2f")
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f")

        # Histórico Alimentar – col2
        with col2:
            family_history = st.selectbox("Histórico Familiar de Sobrepeso?", ["Sim", "Não"])
            favc = st.selectbox(
                "Consome alimentos calóricos com frequência (FAVC)?", ["Sim", "Não"]
            )
            fcvc = st.slider("Frequência de consumo de vegetais (FCVC)", 1, 3, 2)
            ncp = st.slider("Número de refeições principais (NCP)", 1, 4, 3)
            caec = st.selectbox(
                "Come entre as refeições (CAEC)?", ["Não", "Às vezes", "Frequentemente", "Sempre"]
            )

        # Outros Hábitos – col3
        with col3:
            smoke = st.selectbox("Fuma (SMOKE)?", ["Sim", "Não"])
            ch2o = st.slider("Consumo de água diário (litros) (CH2O)", 1, 3, 2)
            scc = st.selectbox("Monitora calorias (SCC)?", ["Sim", "Não"])
            faf = st.slider("Frequência de atividade física (dias/semana) (FAF)", 0, 3, 1)
            tue = st.slider("Tempo de uso de tecnologia (horas/dia) (TUE)", 0, 2, 1)
            calc = st.selectbox(
                "Consumo de álcool (CALC)?", ["Não", "Às vezes", "Frequentemente", "Sempre"]
            )
            mtrans = st.selectbox(
                "Meio de transporte (MTRANS)",
                ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"],
            )

    # ------------------------------------------------------------------
    # Predição quando usuário clica no botão
    # ------------------------------------------------------------------
    if st.button("**Prever Nível de Obesidade**", use_container_width=True):
        input_data = {
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans,
            # Serão adicionadas dinamicamente abaixo as chaves de histórico familiar
        }

        # Compatibilidade de nomes de coluna de histórico familiar
        history_keys = []
        if hasattr(model, "feature_names_in_"):
            if "family_history" in model.feature_names_in_:
                history_keys.append("family_history")
            if "family_history_with_overweight" in model.feature_names_in_:
                history_keys.append("family_history_with_overweight")
        else:
            # fallback – coloca ambas
            history_keys = ["family_history", "family_history_with_overweight"]

        for key in history_keys:
            input_data[key] = family_history

        # Tradução para inglês
        input_data = traduzir_respostas_para_ingles(input_data)

        # DataFrame + Feature Engineering
        input_df = pd.DataFrame([input_data])
        input_df["IMC"] = input_df["Weight"] / (input_df["Height"] ** 2)

        # Verificação de colunas esperadas
        if hasattr(model, "feature_names_in_"):
            missing_cols = set(model.feature_names_in_) - set(input_df.columns)
            if missing_cols:
                st.error(
                    f"As seguintes colunas estão faltando para o modelo '{selected_model_name}': {missing_cols}"
                )
                st.stop()

        # Predição
        prediction = model.predict(input_df)[0]
        prediction_pt = traduzir_predicao_para_portugues(prediction.replace("_", " "))

        # Exibição do resultado
        st.subheader("Resultado da Predição", divider="blue")
        if "Obesity" in prediction or "Obesidade" in prediction_pt:
            st.error(f"**{prediction_pt}**")
        elif "Overweight" in prediction or "Sobrepeso" in prediction_pt:
            st.warning(f"**{prediction_pt}**")
        else:
            st.success(f"**{prediction_pt}**")

# =====================================================================
# ABA 2 – DASHBOARD POWER BI
# =====================================================================
with abas[1]:
    st.subheader("Dashboard Power BI")
    powerbi_url = (
        "https://app.powerbi.com/view?r=eyJrIjoiMjE3NGM2YjEtZjc1Zi00Y2RhLTg5MjctMDhkY2IwNThmNjczIiwidCI6Ijk4ZmM5YWY2LWZkOWItNGI5Yi1hZjA2LTNiY2VjYmQwNzNkMiIsImMiOjR9"
    )
    st.markdown(
        f"""
        <iframe title='Power BI' width='100%' height='600' src='{powerbi_url}' frameborder='0' allowFullScreen='true'></iframe>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# RODAPÉ
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; color: #444; font-size: 1em; margin-bottom: 8px;'>
        <strong>Sobre o Grupo</strong><br>
        Alessandra Barcelos (RM 360.512)<br>
        Henrique de Paulo Gonçalves (RM 359.815)<br>
        Júlio Henrique Nachbar (RM 360.201)<br>
        Rafael Pompeu (RM 359.924)<br>
        Samuel Lima (RM 360.354)
    </div>
    <hr>
    <div style='text-align:center; color: #888; font-size: 0.9em; margin-top: 24px;'>
        Grupo<br>
        <a href='https://github.com/rappantoja' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
