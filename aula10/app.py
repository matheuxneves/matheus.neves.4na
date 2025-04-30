import streamlit as st
from transformers import pipeline

# Título do aplicativo
st.title("🎬 Chatbot de Análise de Filmes")

# Descrição
st.write("Olá! Eu sou o Chatbot de Análise de Filmes. Me conte o que você achou do filme!")

# Entrada do usuário: comentário sobre filme
review = st.text_area("💬 Escreva aqui sua crítica ou opinião sobre o filme:")

# Carregando o modelo de análise de sentimentos
@st.cache_resource
def load_model():
    return pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None)

model = load_model()

# Botão para executar a análise
if st.button("🔍 Analisar Sentimento"):
    if len(review.strip()) > 0:
        # Realizando a análise de sentimentos
        result = model(review)

        # Corrigindo o acesso ao resultado
        sentiment = result[0][0]['label']  # Acessando o primeiro item da lista
        confidence = round(result[0][0]['score'] * 100, 2)

        # Respostas baseadas no sentimento
        if sentiment == 'POSITIVE':
            bot_response = f"Que bom que você gostou! 😄 Acredito que você realmente se conectou com o filme, com uma confiança de {confidence}%. Você tem mais alguma recomendação? fale com os nossos amigos ai mano https://t.me/cinefecafbot"
        elif sentiment == 'NEGATIVE':
            bot_response = f"""
            Sinto muito que não tenha gostado do filme. 😞 Sua opinião é importante! Com uma confiança de {confidence}%.
            Parece que a experiência não foi positiva. Se puder, nos conte mais sobre o que não funcionou para você. Vamos tentar melhorar na próxima! fale com os nossos amigos ai mano https://t.me/cinefecafbot
            """
        else:
            bot_response = f"Você parece estar em cima do muro. 😅 O sentimento é neutro, com uma confiança de {confidence}%. Talvez o filme tenha sido apenas 'ok' para você, né? O que poderia ter melhorado? fale com os nossos amigos ai mano https://t.me/cinefecafbot"

        # Exibindo o resultado de forma amigável
        st.subheader("Resultado da Análise:")
        st.write(bot_response)

        # Exibindo gráfico de barras com as pontuações de sentimento
        sentiment_scores = {
            'POSITIVE': result[0][0]['score'] if sentiment == 'POSITIVE' else 0,
            'NEGATIVE': result[0][0]['score'] if sentiment == 'NEGATIVE' else 0,
            'NEUTRAL': result[0][0]['score'] if sentiment == 'NEUTRAL' else 0
        }
        st.bar_chart(sentiment_scores)
    else:
        st.warning("Por favor, insira sua crítica ou comentário sobre o filme para que eu possa analisá-la.")