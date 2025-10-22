import streamlit as st

# Derlenmiş workflow uygulamamızı ve GraphState modelini alıyoruz.
from graph.graph import app
from graph.state import GraphState


# Sayfa başlığını ve düzenini ayarlıyoruz.
st.set_page_config(page_title="Film Rehberi — Türkçe", layout="centered")


def _init_session():
    # Streamlit sayfa yenilendiğinde konuşma geçmişini ve input alanını
    # session_state içinde saklıyoruz. Böylece kullanıcı geçmişi korunur.
    if "messages" not in st.session_state:
        # Uygulama açıldığında asistanın karşılama mesajını hazır tutuyoruz.
        st.session_state.messages = [
            {"role": "assistant", "text": "Merhaba! Film sorularınızda yardımcı olabilirim. Hangi film hakkında bilgi istersiniz?"}
        ]
    if "user_input" not in st.session_state:
        # Kullanıcının yazdığı metni tutacak alan.
        st.session_state.user_input = ""


_init_session()


st.title("Film Rehberi — Sohbet")
st.markdown("Film önerileri ve kullanıcı yorumları hakkında sohbet edebilirsiniz. Birine mesaj atıyormuş gibi yazın.")


def send_message():
    # Kullanıcı input'unu alıp boş değilse graf akışını tetikleriz.
    ui = st.session_state.user_input.strip()
    if not ui:
        return

    # Kullanıcı mesajını geçmişe ekle.
    st.session_state.messages.append({"role": "user", "text": ui})

    try:
        # Workflow'a yeni bir GraphState gönderip sonucu alıyoruz.
        result = app.invoke(GraphState(message=ui))
        # Öncelikle 'answer' alanına bak; yoksa 'context' dönebilir; ikisi de yoksa hata mesajı göster.
        resp = result.get("answer") or result.get("context") or "Maalesef cevap üretilmedi."
    except Exception as e:
        # Herhangi bir hata durumunu kullanıcıya gösteriyoruz (geliştirme aşamasında faydalı).
        resp = f"Hata: {e}"

    # Asistanın cevabını geçmişe ekle ve input alanını temizle.
    st.session_state.messages.append({"role": "assistant", "text": resp})
    st.session_state.user_input = ""


# Konuşma geçmişini ekranda gösteriyoruz. 'role' alanına göre başlık değişir.
for msg in st.session_state.messages:
    role = "Siz" if msg["role"] == "user" else "Asistan"
    st.markdown(f"**{role}:** {msg['text']}")


# Metin girişi alanı. Kullanıcı enter'e bastığında send_message tetiklenir.
st.text_input(
    "Mesajınızı yazın...",
    key="user_input",
    on_change=send_message,
)
