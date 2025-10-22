# Bu dosya, kullanıcının mesajını hangi 'intent'e (niyet) yönlendireceğimizi belirler.
# Yani kullanıcı film hakkında mı soru soruyor yoksa genel sohbet mi etmek istiyor,
# ona karar veren küçük bir sınıflandırıcıyı tanımlıyoruz.

# OpenRouter üzerinden tanımlı LLM nesnesini alıyoruz; bu LLM'i intent sınıflandırması
# ve daha sonra yapılandırılmış çıktı almak için kullanacağız.
from open_router import llm

# Literal tipiyle intent alanının yalnızca iki olası değer almasını sağlıyoruz.
from typing import Literal

# Chat prompt'larını daha kolay kurmak için LangChain'in ChatPromptTemplate yardımcı sınıfını kullanıyoruz.
from langchain_core.prompts import ChatPromptTemplate

# Pydantic ile yapılandırılmış (typed) çıktı modelini tanımlamak için gerekli sınıflar.
from pydantic import BaseModel, Field


class RouteIntent(BaseModel):
    # Bu model, LLM'den beklediğimiz yapılandırılmış çıktının şemasını tanımlar.
    # LLM'den sadece {'intent': 'film_query' } veya {'intent': 'general_chat'}
    # şeklinde kısa JSON dönmesini bekliyoruz.
    """Kullanıcının mesajını hangi intent'e (film_query veya general_chat) yönlendirdiğimizi tanımlar."""
    intent: Literal[
        "film_query",
        "general_chat"
    ] = Field(
        ...,
        description=(
            # Alan açıklaması: kullanıcı mesajı film ile ilgiliyse 'film_query', değilse 'general_chat'
            "film_query → Film, oyuncu, yönetmen, puan, yorum veya tavsiye soruları\n"
            "general_chat → Selamlaşma veya film dışı sohbet"
        )
    )


# LLM üzerinde yukarıdaki Pydantic modelini kullanarak yapılandırılmış çıktı (structured output)
# isteğinde bulunabilmek için helper oluştuyoruz. Bu, LLM cevabını otomatik olarak
# RouteIntent modeline parse etmeye çalışacak.
structured_intent_router = llm.with_structured_output(RouteIntent)


# Sistem prompt'u: LLM'ye nasıl davranacağını ve hangi kurallara uyacağını söylüyoruz.
# Burada özellikle sadece tek bir intent değeri döndürmesini ve ekstra metin üretmemesini
# istiyoruz. Ayrıca film ile ilgili şüpheli durumlarda film_query seçilmesi gerektiği vurgulanıyor.
system = """
Sen bir intent sınıflandırıcısın.
Kullanıcının mesajını oku ve aşağıdaki iki intentten **tam olarak birini** seç.

KURALLAR:
- Sadece `film_query` veya `general_chat` döndür.
- JSON formatında şu şekilde döndür: {{"intent": "<seçilen_intent>"}}
- Açıklama, neden veya fazladan metin ekleme.

TANIMLAR:
- film_query:
  • Film adı, oyuncu, yönetmen, puan, yorum, tavsiye, sinema, dizi hakkında HERHANGI bir soru
  • Film önerisi istekleri (komedi, aksiyon, dram vb.)
  • Belirli filmlerin değerlendirmeleri
  • "Ne izlemeliyim?", "Film öner", "Hangi film iyi?" gibi sorular
  • Örnekler:
    - "Avatar filmi nasıldı?"
    - "İyi bir bilim kurgu önerir misin?"
    - "Bugün ne izlememi önerirsin?"
    - "Komedi modundayım"
    - "James Cameron hangi filmleri yönetti?"
    - "Avatar yorumları nasıl?"
- general_chat:
  • Sadece selamlaşma, hal hatır, genel sohbet
  • Film ile hiçbir ilgisi olmayan konular
  • Örnekler:
    - "Merhaba nasılsın?"
    - "Bugün hava nasıl?"
    - "Teşekkürler"

ÖNEMLİ: Şüpheli durumlarda film_query seç. Film kelimesi geçmese bile eğlence/izleme ile ilgiliyse film_query seç.

Yalnızca seçtiğin intent adını içeren JSON döndür.
"""


# ChatPromptTemplate, system mesajını ve kullanıcı mesajını alıp tek bir prompt oluşturuyor.
# Burada human kısımdaki {question} placeholder'ı çağıran kod tarafından doldurulacak.
intent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system.strip()),
        ("human", "{question}")
    ]
)


# Son olarak prompt ile structured output router'ı zincirleyerek tek bir "question_router"
# oluşturuyoruz. Bu nesne invoke edildiğinde önce prompt hazırlanır, sonra LLM'den
# yapılandırılmış (RouteIntent) JSON bekler.
question_router = intent_prompt | structured_intent_router