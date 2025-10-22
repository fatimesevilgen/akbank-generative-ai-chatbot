"""
Bu node, verilen `state` içeriğine göre LLM'den (open_router.llm) bir
yanıt üretir ve hem `answer` hem de `generation` alanlarını doldurur.

Davranış:
- `context` varsa prompt içinde bağlam olarak verilir.
- `message` kullanıcı isteğidir; prompt içerisinde soru olarak yer alır.
- Eğer LLM çağrısı başarısız olursa, hata mesajını `answer` alanına yazar.
"""

# LLM adapter'ını içe aktarıyoruz. open_router projesindeki llm wrapper'ı
# invoke/metodu ile prompt'a yanıt almayı sağlar.
from open_router import llm


def generate(state):
    """LLM'den yanıt üretir.

    Args:
        state: GraphState veya benzeri, `message` ve opsiyonel `context` alanları içerir.

    Returns:
        state dict'i güncellenmiş `answer` ve `generation` alanları ile.
    """

    # Eğer context None ise boş string ile değiştirelim (prompt'a güvenli şekilde eklemek için).
    context = state.context or ""
    message = state.message.strip()

    # Prompt'u hazırlıyoruz: kullanıcının sorusu ve varsa bağlam burada sunulur.
    prompt = f"""
Sen bir film öneri asistanısın.
Aşağıda sana filmle ilgili bağlam bilgileri verilmiştir.

Soru: {message}

Bağlam:
{context}

Eğer filmle ilgili bilgiler verilmişse, bu bilgiler üzerinden net ve kısa bir yanıt ver.
Eğer verilmemişse, "film bulunamadı" deme; sadece "Bu film hakkında bilgi bulunamadı" şeklinde kibar bir yanıt ver.
"""

    try:
        # LLM'i çağır ve cevabı al
        response = llm.invoke(prompt)
        # Bazı adapterlar direkt content attribute döner, bazıları farklı yapıda olabilir.
        # getattr ile hem `.content` varsa onu kullanıyoruz, yoksa objeyi string'e çeviriyoruz.
        answer = getattr(response, "content", str(response)).strip()

        # Hem generation (ara üretim) hem de answer alanını aynı metinle dolduruyoruz.
        return {**state.dict(), "answer": answer, "generation": answer}
    except Exception as e:
        # Hata durumunda kullanıcıya anlaşılır bir hata mesajı döneriz; generation boş bırakılır.
        return {
            **state.dict(),
            "answer": f"Bir hata oluştu: {e}",
            "generation": ""
        }
