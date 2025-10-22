# typing modülünden gelen yardımcı tipler:
# - Optional: bir alanın None olabileceğini belirtmek için kullanılır.
# - List: liste tip tanımlamaları yapmak için kullanılır.
# - Literal: belirli sabit değerler (ör. 'film_query') ile tip sınırlandırması sağlar.
from typing import Optional, List, Literal

# Pydantic'in BaseModel sınıfı, veri doğrulama, tip dönüştürme ve daha
# okunabilir modeller oluşturmak için kullanılır. GraphState Pydantic modelidir.
from pydantic import BaseModel

# langchain_core.documents.Document: LangChain içinde döküman/unik içerik
# taşıyan nesne tipi. Retrieval çıktıları genellikle Document objeleri olur.
from langchain_core.documents import Document


class GraphState(BaseModel):
        """
        Graph akışı boyunca taşınan durum (state) objesi.

        Açıklama:
        - Bu Pydantic modeli, workflow içindeki farklı node'lar arasında paylaşılan
            veriyi tek bir yerde toplar. Her node, bu nesnedeki alanları okuyup
            güncelleyerek sonraki adımların kullanacağı veriyi hazırlar.

        Alanlar (kısaca):
        - message: Kullanıcıdan gelen orijinal metin/girdi.
        - intent: Modelin çıkardığı niyet (örn. film sorgusu veya genel sohbet).
        - retrieved_docs: RAG (retrieval-augmented generation) için getirilmiş Document listesi.
        - context: Dokümanlardan veya önceki adımlardan oluşturulmuş ek bağlam metni.
        - generation: Modelin ürettiği ara metin (ör. özet, genişleme).
        - answer: Nihai cevap (kullanıcıya dönecek metin).
        - history: İşlem boyunca kaydedilen kısa geçmiş kayıtları (liste olarak tutulur).

        Notlar:
        - Bu sınıf, Graph tabanlı bir pipeline içinde tek bir state objesinin
            paylaşılması için düşünülmüştür. Alanlar opsiyonel tutulmuş, böylece sadece
            gerekli olanlar doldurulur.
        """
        # Kullanıcıdan gelen veya Ön yüz tarafından set edilen mesaj.
        message: str

        # Çıkarılmış intent: uygulama mantığına göre belirlenmiş literal değerler.
        intent: Optional[Literal["film_query", "general_chat"]] = None

        # Bir liste halinde getirilen dokümanlar (langchain.documents.Document nesneleri).
        # RAG senaryosunda kullanılan retrieval çıktıları burada tutulur.
        retrieved_docs: Optional[List[Document]] = None

        # Birleştirilmiş/özetlenmiş bağlam metni. Örneğin getirilen dokümanlardan
        # üretilecek kısa bağlam burada saklanabilir.
        context: Optional[str] = None

        # Modelin ara üretimleri (ör: bir özet, token bazlı üretim) için ayrılmış alan.
        generation: Optional[str] = None

        # Kullanıcıya dönecek nihai cevap metni.
        answer: Optional[str] = None

        # İş akışı boyunca kısa geçmiş kayıtları. Varsayılan boş liste, ancak
        # dikkat: mutable default olmasına dikkat (gerekirse factory kullanılabilir).
        history: list = []

        class Config:
                # Document gibi Pydantic'in bilmediği tipleri kabul etmesini sağlar.
                arbitrary_types_allowed = True
