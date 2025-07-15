# --- 1. Importação das Bibliotecas ---
import streamlit as st
import pandas as pd
import joblib

# --- HEADER ---
st.markdown(
    """
    <div style='background-color:#0d6efd;padding:16px;border-radius:8px;margin-bottom:16px'>
        <h2 style='color:white;text-align:center;margin:0;'>Bem-vindo ao Sistema Preditivo de Obesidade</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# --- TABS ---
abas = st.tabs(["Predição", "Dashboard Power BI"])

with abas[0]:
    # Função para traduzir respostas do questionário do pt-br para inglês (esperado pelo modelo)
    def traduzir_respostas_para_ingles(input_data):
        traducoes = {
            'Gender': {'Masculino': 'Male', 'Feminino': 'Female', 'Male': 'Male', 'Female': 'Female'},
            'family_history': {'Sim': 'yes', 'Não': 'no', 'yes': 'yes', 'no': 'no'},
            'family_history_with_overweight': {'Sim': 'yes', 'Não': 'no', 'yes': 'yes', 'no': 'no'},
            'FAVC': {'Sim': 'yes', 'Não': 'no', 'yes': 'yes', 'no': 'no'},
            'CAEC': {
                'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always',
                'no': 'no', 'Sometimes': 'Sometimes', 'Frequently': 'Frequently', 'Always': 'Always'
            },
            'SMOKE': {'Sim': 'yes', 'Não': 'no', 'yes': 'yes', 'no': 'no'},
            'SCC': {'Sim': 'yes', 'Não': 'no', 'yes': 'yes', 'no': 'no'},
            'CALC': {
                'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always',
                'no': 'no', 'Sometimes': 'Sometimes', 'Frequently': 'Frequently', 'Always': 'Always'
            },
            'MTRANS': {
                'Transporte Público': 'Public_Transportation', 'Automóvel': 'Automobile', 'Caminhada': 'Walking',
                'Motocicleta': 'Motorbike', 'Bicicleta': 'Bike',
                'Public_Transportation': 'Public_Transportation', 'Automobile': 'Automobile',
                'Walking': 'Walking', 'Motorbike': 'Motorbike', 'Bike': 'Bike'
            }
        }
        for campo, valor in input_data.items():
            if campo in traducoes and valor in traducoes[campo]:
                input_data[campo] = traducoes[campo][valor]
        return input_data

    # Função para traduzir o resultado da predição do inglês para pt-br
    def traduzir_predicao_para_portugues(prediction_text):
        traducoes = {
            'Insufficient Weight': 'Peso Insuficiente',
            'Normal Weight': 'Peso Normal',
            'Overweight Level I': 'Sobrepeso Nível I',
            'Overweight Level II': 'Sobrepeso Nível II',
            'Obesity Type I': 'Obesidade Tipo I',
            'Obesity Type II': 'Obesidade Tipo II',
            'Obesity Type III': 'Obesidade Tipo III'
        }
        return traducoes.get(prediction_text, prediction_text)

    # --- 2. Carregamento do Modelo Treinado ---
    @st.cache_resource
    def load_model(model_path):
        """Carrega o pipeline de machine learning salvo."""
        return joblib.load(model_path)

    # --- 3. Configuração da Página e Título ---
    st.set_page_config(page_title="Sistema Preditivo de Obesidade", layout="wide")

    st.title('🩺 Sistema Preditivo para Níveis de Obesidade')
    st.markdown("""
    Esta ferramenta foi desenvolvida para apoiar a equipe médica no diagnóstico de níveis de obesidade[cite: 15, 19].
    Utilizando um modelo de Machine Learning, o sistema analisa as informações do paciente para fornecer uma predição precisa.
    **Instruções:** Preencha os campos abaixo com os dados do paciente e clique em 'Prever' para obter o resultado.
    """)
    # Adicionando um aviso importante
    st.info("**Aviso:** Esta é uma ferramenta de apoio à decisão e não substitui o diagnóstico clínico realizado por um profissional de saúde qualificado.")

    # --- Escolha do Modelo ---
    model_options = {
        "KNN": "app_streamlit/models/KNN.pkl",
        "Random Forest": "app_streamlit/models/RandomForest.pkl",
        "SVM": "app_streamlit/models/SVM.pkl"
    }
    st.sidebar.header("Configuração do Modelo")
    selected_model_name = st.sidebar.selectbox("Selecione o modelo de Machine Learning", list(model_options.keys()), index=1)
    selected_model_path = model_options[selected_model_name]

    # Tenta carregar o modelo e exibe mensagem de erro se não encontrar o arquivo
    try:
        model = load_model(selected_model_path)
    except FileNotFoundError:
        st.error(f"Arquivo do modelo não encontrado: '{selected_model_path}'.\n\nVerifique se o arquivo existe no diretório correto.")
        st.stop()

    # --- 4. Coleta de Dados do Usuário com a Interface ---
    # Usamos um container para agrupar os campos do formulário
    with st.container():
        st.header("Formulário de Dados do Paciente")
        
        # Layout em colunas para uma melhor organização dos campos de entrada
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Informações Pessoais")
            age = st.number_input('Idade', min_value=1, max_value=100, value=25, help="Idade do paciente [cite: 22]")
            gender = st.selectbox('Gênero', ['Masculino', 'Feminino'], help="Gênero do paciente [cite: 21]")
            height = st.number_input('Altura (metros)', min_value=1.0, max_value=2.5, value=1.70, format="%.2f", help="Altura em metros [cite: 23]")
            weight = st.number_input('Peso (kg)', min_value=30.0, max_value=200.0, value=70.0, format="%.1f", help="Peso em quilogramas [cite: 24]")

        with col2:
            st.subheader("Histórico e Hábitos Alimentares")
            family_history = st.selectbox('Histórico Familiar de Sobrepeso?', ['Sim', 'Não'], help="Se algum membro da família sofre de sobrepeso [cite: 25]")
            favc = st.selectbox('Consome alimentos calóricos com frequência (FAVC)?', ['Sim', 'Não'], help="Consumo frequente de alimentos altamente calóricos [cite: 26]")
            fcvc = st.slider('Frequência de consumo de vegetais (FCVC)', 1, 3, 2, help="1: Nunca, 2: Às vezes, 3: Sempre [cite: 27]")
            ncp = st.slider('Número de refeições principais (NCP)', 1, 4, 3, help="Quantas refeições principais faz diariamente [cite: 29]")
            caec = st.selectbox('Come entre as refeições (CAEC)?', ['Não', 'Às vezes', 'Frequentemente', 'Sempre'], help="Se come algo entre as refeições [cite: 30]")

        with col3:
            st.subheader("Outros Hábitos de Vida")
            smoke = st.selectbox('Fuma (SMOKE)?', ['Sim', 'Não'], help="Se o paciente é fumante [cite: 30]")
            ch2o = st.slider('Consumo de água diário (litros) (CH2O)', 1, 3, 2, help="1: <1L, 2: 1-2L, 3: >2L [cite: 31]")
            scc = st.selectbox('Monitora calorias (SCC)?', ['Sim', 'Não'], help="Se monitora as calorias que ingere diariamente [cite: 32]")
            faf = st.slider('Frequência de atividade física (dias/semana) (FAF)', 0, 3, 1, help="0: Nenhuma, 1: 1-2 dias, 2: 2-4 dias, 3: 4-5 dias [cite: 34]")
            tue = st.slider('Tempo de uso de tecnologia (horas/dia) (TUE)', 0, 2, 1, help="0: 0-2h, 1: 3-5h, 2: >5h [cite: 36, 37]")
            calc = st.selectbox('Consumo de álcool (CALC)?', ['Não', 'Às vezes', 'Frequentemente', 'Sempre'], help="Frequência de consumo de álcool [cite: 38]")
            mtrans = st.selectbox('Meio de transporte (MTRANS)', ['Transporte Público', 'Automóvel', 'Caminhada', 'Motocicleta', 'Bicicleta'], help="Meio de transporte que costuma usar [cite: 39]")


    # --- 5. Botão e Lógica de Predição ---
    # O botão de predição, quando clicado, aciona o modelo
    if st.button('**Prever Nível de Obesidade**', use_container_width=True):
        # Garante compatibilidade de nomes de coluna para todos os modelos
        # Se o modelo espera 'family_history', use esse nome
        # Se espera 'family_history_with_overweight', use esse nome
        # Se espera ambos, envie ambos
        family_history_keys = []
        if hasattr(model, 'feature_names_in_'):
            if 'family_history' in model.feature_names_in_:
                family_history_keys.append('family_history')
            if 'family_history_with_overweight' in model.feature_names_in_:
                family_history_keys.append('family_history_with_overweight')
        else:
            # fallback: mantém ambos
            family_history_keys = ['family_history', 'family_history_with_overweight']

        input_data = {
            'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
            'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp,
            'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf,
            'TUE': tue, 'CALC': calc, 'MTRANS': mtrans
        }
        for key in family_history_keys:
            input_data[key] = family_history

        # Traduz as respostas para inglês antes de enviar ao modelo
        input_data = traduzir_respostas_para_ingles(input_data)

        input_df = pd.DataFrame([input_data])
        input_df['IMC'] = input_df['Weight'] / (input_df['Height']**2)

        # Garante que todas as colunas esperadas pelo modelo estejam presentes
        if hasattr(model, 'feature_names_in_'):
            expected_cols = set(model.feature_names_in_)
            input_cols = set(input_df.columns)
            missing = expected_cols - input_cols
            if missing:
                st.error(f"As seguintes colunas estão faltando para o modelo '{selected_model_name}': {missing}")
                st.stop()

        # Utiliza o modelo carregado para fazer a predição
        prediction = model.predict(input_df)
        
        # --- 6. Exibição do Resultado ---
        st.subheader('Resultado da Predição', divider='blue')
        
        # Formata o texto da predição para ser mais legível
        prediction_text = prediction[0].replace("_", " ")
        # Traduz o resultado para português
        prediction_text_pt = traduzir_predicao_para_portugues(prediction_text)
        
        # Usa cores diferentes para destacar o nível de severidade do resultado
        if "Obesity" in prediction_text or "Obesidade" in prediction_text_pt:
            st.error(f'**{prediction_text_pt}**')
        elif "Overweight" in prediction_text or "Sobrepeso" in prediction_text_pt:
            st.warning(f'**{prediction_text_pt}**')
        else:
            st.success(f'**{prediction_text_pt}**')

with abas[1]:
    st.subheader("Dashboard Power BI")
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiMjE3NGM2YjEtZjc1Zi00Y2RhLTg5MjctMDhkY2IwNThmNjczIiwidCI6Ijk4ZmM5YWY2LWZkOWItNGI5Yi1hZjA2LTNiY2VjYmQwNzNkMiIsImMiOjR9"  # Substitua pelo seu link de incorporação
    st.markdown(
        f"""
        <iframe title="Power BI" width="100%" height="600" src="{powerbi_url}" frameborder="0" allowFullScreen="true"></iframe>
        """,
        unsafe_allow_html=True
    )

# --- FOOTER ---
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
    unsafe_allow_html=True
)