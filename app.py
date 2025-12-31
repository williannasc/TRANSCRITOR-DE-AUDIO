import streamlit as st
import whisper
import os
import tempfile
from fpdf import FPDF

# 1. Configuração de Estilo e Página
st.set_page_config(page_title="Transcritor Master", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    .stTextArea textarea { font-family: 'serif'; font-size: 18px; line-height: 1.5; border-radius: 15px; background-color: #1e1e1e; color: #ffffff; }
    .stButton>button { border-radius: 25px; height: 3em; background-color: #1DB954; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #1ed760; border: none; }
    </style>
    """, unsafe_allow_html=True)

# 2. Funções de Suporte
@st.cache_resource
def carregar_modelo(modelo):
    return whisper.load_model(modelo)

def gerar_pdf(titulo, artista, texto):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 20)
    pdf.cell(0, 15, titulo.upper(), ln=True, align='C')
    pdf.set_font("helvetica", 'I', 14)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, artista, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("helvetica", size=12)
    pdf.set_text_color(0, 0, 0)
    for linha in texto.split('\n'):
        pdf.cell(0, 7, linha, ln=True, align='C')
    return pdf.output()

# 3. Barra Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Opções")
    nome_musica = st.text_input("Título da Música", "Título da Música")
    nome_artista = st.text_input("Artista", "Cantor Desconhecido")
    modelo_tipo = st.selectbox("Precisão (Modelo)", ["tiny", "base", "small", "medium", "turbo", "large-v3"], index=2)
    pausa = st.slider("Sensibilidade de Estrofe (s)", 1.0, 5.0, 2.5)
    st.divider()
    st.info("O modelo 'small' é o melhor custo-benefício entre velocidade e precisão.")
    st.info("Ajuste a sensibilidade de estrofe para controlar quebras na letra transcrita. Quanto maior o numero, mais espaçamento entre estrofes.")

# 4. Interface Principal
st.title("🎵 Transcritor de Letras de Músicas")

col_esq, col_dir = st.columns([1, 1], gap="large")

with col_esq:
    st.subheader("📤 Upload e Player")
    arquivo = st.file_uploader("Arraste seu áudio (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
    
    if arquivo:
        # Re-inserindo o Player que você gostou!
        st.audio(arquivo)
        
        if st.button("🚀 Iniciar Transcrição"):
            with st.spinner("Analisando áudio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(arquivo.getvalue())
                    path = tmp.name
                
                try:
                    model = carregar_modelo(modelo_tipo)
                    # Usando os parâmetros de precisão que discutimos antes
                    res = model.transcribe(
                        path, 
                        language="pt", 
                        temperature=0, 
                        condition_on_previous_text=False
                    )
                    
                    linhas = []
                    fim_ant = 0.0
                    for s in res["segments"]:
                        if s["start"] - fim_ant > pausa:
                            linhas.append("")
                        linhas.append(s["text"].strip())
                        fim_ant = s["end"]
                    
                    st.session_state['letra_final'] = "\n".join(linhas)
                finally:
                    os.remove(path)

with col_dir:
    st.subheader("📄 Letra e Impressão")
    if 'letra_final' in st.session_state:
        # Área de edição com fonte maior para conferência
        letra_editada = st.text_area(
            "Edite o texto abaixo para ajustar detalhes:", 
            st.session_state['letra_final'], 
            height=450
        )
        
        # Botões de ação em colunas menores
        b1, b2 = st.columns(2)
        with b1:
            # Geramos o PDF (que retorna um bytearray)
            pdf_output = gerar_pdf(nome_musica, nome_artista, letra_editada)
            
            # CONVERSÃO: Transformamos bytearray em bytes para o Streamlit aceitar
            pdf_bytes = bytes(pdf_output)
            
            st.download_button(
                label="📥 Baixar PDF Pronto", 
                data=pdf_bytes, 
                file_name=f"{nome_musica}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with b2:
            if st.button("🗑️ Limpar", use_container_width=True):
                del st.session_state['letra_final']
                st.rerun()
    else:
        st.info("A letra aparecerá aqui após a transcrição.")

st.divider()
st.caption("Desenvolvido com OpenAI Whisper & Streamlit por wn.dev.br")