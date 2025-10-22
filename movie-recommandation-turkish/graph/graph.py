# Bu dosya, uygulamanın akış (workflow) grafını tanımlar.
# Yani kullanıcı mesajı geldiğinde hangi düğümlerin (nodes) hangi sırayla çalışacağını belirler.

# StateGraph ve END sabitini langgraph kütüphanesinden alıyoruz. StateGraph, düğümleri ve
# geçişleri (edges) tanımlamamızı sağlayan ana sınıftır. END grafın sonunu işaret eder.
from langgraph.graph import StateGraph, END

# GraphState, graf boyunca taşınacak durum nesnesinin (state) tipini tanımlar.
from graph.state import GraphState

# Route dosyasında tanımladığımız question_router; kullanıcı mesajından intent'i
# çıkaran (film_query vs general_chat) LLM tabanlı sınıflandırıcıdır.
from graph.chains.route import question_router

# Grafın düğümleri: veri getirme, yanıt üretme ve genel sohbet için tanımlanmış fonksiyonlar.
from graph.nodes.retrieve import movie_retrieve
from graph.nodes.generate import generate
from graph.nodes.general_chat import general_chat

# intent.py içinde iki sabit tanımlı: FILM_QUERY ve GENERAL_CHAT. Bunları burada kullanacağız.
from intent import FILM_QUERY, GENERAL_CHAT


def detect_intent(state: GraphState):
    # Bu fonksiyon grafın giriş noktasında çağrılır.
    # state.message içindeki kullanıcı mesajını alır, question_router ile LLM'e gönderir
    # ve dönen yapılandırılmış sonuçtan (RouteIntent) intent değerini alıp döndürür.
    out = question_router.invoke({"question": state.message})
    return out.intent


# StateGraph'in örneğini oluşturuyoruz; GraphState tipini her node'un alacağı durum nesnesi
# olarak belirtiyoruz.
workflow = StateGraph(GraphState)


# Grafın düğümlerini ekliyoruz. Her düğüm bir fonksiyon referansı alır ve çağrıldığında
# GraphState'i alıp güncelleyerek yeni alanlar ekleyebilir (ör. context, answer).
workflow.add_node("MovieRetrieve", movie_retrieve)
workflow.add_node("Generate", generate)
workflow.add_node("GeneralChat", general_chat)


# Giriş noktasını koşullu hale getiriyoruz: detect_intent fonksiyonu hangi intent'i
# döndürürse graf o intent'e karşılık gelen düğümü başlatacak.
workflow.set_conditional_entry_point(
    detect_intent,
    {
        FILM_QUERY: "MovieRetrieve",
        GENERAL_CHAT: "GeneralChat",
    }
)


# Noktalar arası geçişleri tanımlıyoruz: Film verisi getirildikten sonra `Generate` çalışsın,
# `Generate` veya `GeneralChat` tamamlandığında ise akış sonlansın (END).
workflow.add_edge("MovieRetrieve", "Generate")
workflow.add_edge("Generate", END)
workflow.add_edge("GeneralChat", END)


# Grafı derleyip (compile) kullanıma hazır hale getiriyoruz; `app` bu derlenmiş workflow objesidir.
app = workflow.compile()