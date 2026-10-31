import json
import os
import dspy
import pandas as pd
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(page_title="E-Ticaret Asistanı ", page_icon="🛍️ ", layout="centered")

st.title("🛍️ Akıllı E-Ticaret Destek Asistanı")
st.write("Sipariş ve stok durumunuzu anında sorgulayan akıllı DSPy ajanı.")

# DSPy Yapılandırması
@st.cache_resource
@st.cache_resource
def init_agent():
    # Doğrudan terminalden verdiğimiz çevre değişkenini okuyoruz
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("Lütfen terminalde GEMINI_API_KEY tanımladığınızdan emin olun!")

    gemini_lm = dspy.LM("gemini/models/gemini-3.5-flash", api_key=api_key)
    dspy.configure(lm=gemini_lm)

    def kargo_durumu_sorgula(siparis_no: str) -> str:
        """Müşterinin sipariş numarasına göre kargo ve iade durumunu sorgular."""
        df = pd.read_csv("siparisler.csv")
        sonuc = df[df["siparis_no"].astype(str) == str(siparis_no)]
        if not sonuc.empty:
            return json.dumps(sonuc.to_dict(orient="records")[0], ensure_ascii=False)
        return json.dumps({"hata": f"{siparis_no} numaralı sipariş bulunamadı."})

    def stok_model_kontrolu(urun_kodu_veya_adi: str) -> str:
        """Ürün kodu veya adına göre depodaki stok miktarını ve fiyatını sorgular."""
        df = pd.read_csv("stoklar.csv")
        sonuc = df[df["urun_kodu"].str.lower().str.contains(urun_kodu_veya_adi.lower()) | 
                   df["urun_adi"].str.lower().str.contains(urun_kodu_veya_adi.lower())]
        if not sonuc.empty:
            return json.dumps(sonuc.to_dict(orient="records"), ensure_ascii=False)
        return json.dumps({"hata": f"'{urun_kodu_veya_adi}' için stok kaydı bulunamadı."})

    tools = [kargo_durumu_sorgula, stok_model_kontrolu]
    return dspy.ReAct("question -> answer", tools=tools)

agent = init_agent()

# Chat Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı Girdisi
if prompt := st.chat_input("Örn: 55555 numaralı siparişimin durumu nedir?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Asistan verileri kontrol ediyor..."):
            try:
                yanit = agent(question=prompt)
                response_text = yanit.answer
            except Exception as e:
                response_text = f"Bir hata oluştu: {str(e)}"
        
        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
