"""
Basit bir "general chat" node'u.

Bu dosya iki ana parçadan oluşur:
- `llm` importu: `open_router` modülünden dışarı aktarılan bir LLM
  (örneğin OpenRouter veya başka bir adapter) örneğini alır.
- `general_chat` fonksiyonu: gelen `state` içindeki `message`'ı
  kullanarak LLM'e bir prompt gönderir ve dönen cevabı state'e ekleyerek
  yeni bir dict döner.

Not: Bu node doğrudan `state` objesinin dict temsiline ekleme yapar.
"""

# LLM adapter'ını içe aktarıyoruz. `open_router.llm` projenizde bir LLM
# çağırma wrapper'ıdır; invoke/metodunu kullanarak prompt'a yanıt alır.
from open_router import llm


def general_chat(state):
    """General chat node'u.

    Args:
        state: GraphState veya benzeri, içinde `message` alanı bulunan nesne.

    Döndürür:
        state sözlüğünün güncellenmiş hali (`answer` alanı eklenmiş).
    """

    # Prompt'u oluşturuyoruz. Burada LLM'e, bir film asistanı olduğunu ve
    # kullanıcıyla nazikçe sohbet etmesini söylüyoruz. Eğer kullanıcı film
    # ile ilgili soru soruyorsa, daha spesifik sorular sorması yönünde yönlendirme
    # de ekliyoruz.
    prompt = f"""
Sen yardımcı bir film asistanısın. Kullanıcıyla nazik bir şekilde sohbet et.

Eğer kullanıcı film hakkında soru sorarsa, onları film sorularını daha spesifik şekilde sormaya yönlendir.

Kullanıcı mesajı: {state.message}

Yanıt:"""

    # LLM'i çağırıp cevabı alıyoruz. `invoke` metodunun döndürdüğü objeden
    # `.content` ile düz metin cevabı alıyoruz (open_router adapter'ınıza bağlı).
    ans = llm.invoke(prompt).content

    # Orijinal state'i dict'e çevirip `answer` alanını ekleyerek döndürüyoruz.
    return {**state.dict(), "answer": ans}
