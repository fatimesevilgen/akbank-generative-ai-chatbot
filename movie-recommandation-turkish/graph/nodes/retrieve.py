"""
Bu node, `movie_retriever` aracılığıyla film verisini getirir ve RAG için
uygun bir metin bağlamı (context) oluşturur.

Ana adımlar:
- `query` oluşturma: kullanıcı mesajından temizlenmiş sorgu alınır.
- `movie_retriever.invoke(query)`: embedding tabanlı arama yapılarak ilgili dokümanlar getirilir.
- Eğer sonuç yoksa, kullanıcıya uygun bir uyarı (context) döndürülür.
- Sonuç varsa, duplicate olmaması için (film adı, tip) ikilisiyle filtreleme yapılır
  ve her dokümandan okunabilir bir blok üretilir.
"""

# Vektör veritabanı/retriever'ı içe aktarıyoruz (ingestion.py içinde oluşturulan Chroma retriever).
from ingestion import movie_retriever


def movie_retrieve(state):
    """Film verisini RAG için getirir ve biçimlendirir.

    Args:
        state: GraphState veya benzeri, `message` alanı içeren nesne.

    Returns:
        state dict'i güncellenmiş `retrieved_docs` ve `context` alanları ile.
    """

    # Kullanıcının sorgusunu temizleyip kullanıyoruz.
    query = state.message.strip()

    # Retriever'ı çağırıyoruz. adapter'a göre `.invoke` kullanılıyor.
    docs = movie_retriever.invoke(query)

    # Eğer hiçbir doküman gelmediyse, uygun bir context mesajı döneriz.
    if not docs or len(docs) == 0:
        return {
            **state.dict(),
            "retrieved_docs": [],
            "context": "Veri tabanında ilgili film bulunamadı."
        }

    context_blocks = []
    # Aynı film ve tip kombinasyonunu tekrarlamamak için set ile takip ediyoruz.
    seen = set()

    for doc in docs:
        # Metadata içinden film adı ve doküman tipi (desc veya review vb.) alınır.
        name = doc.metadata.get("name", "Bilinmeyen Film")
        dtype = doc.metadata.get("type", "bilgi")

        # (isim, tip) ile duplicate kontrolü
        key = (name, dtype)
        if key in seen:
            continue
        seen.add(key)

        # Eğer metadata içinde eksik alanlar varsa okunabilir defaultlar veriyoruz.
        genre = doc.metadata.get("genre", "Belirtilmemiş")
        director = doc.metadata.get("directors", "Belirtilmemiş")
        rating = doc.metadata.get("rating", "Puanlanmamış")

        # Her doküman için okunabilir bir blok üretiyoruz; tip 'desc' ise başlık 'Açıklama',
        # değilse 'Kullanıcı Yorumu' olarak etiketlenir.
        block = f"""[Film: {name}]
            Tür: {genre}
            Yönetmen: {director}
            Puan: {rating}
            {"Açıklama" if dtype == "desc" else "Kullanıcı Yorumu"}:
            {doc.page_content}
        ---"""
        context_blocks.append(block)

    # Son olarak state'in dict'ine getirilen dokümanları ve birleştirilmiş context'i ekleyip döndürüyoruz.
    return {
        **state.dict(),
        "retrieved_docs": docs,
        "context": "\n".join(context_blocks)
    }
