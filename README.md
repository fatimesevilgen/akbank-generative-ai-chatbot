# Film Rehberi — Türkçe Film Öneri ve Bilgi Asistanı

Bu proje, Türkçe film açıklamaları ve kullanıcı yorumları kullanarak kullanıcılara film önerileri ve film hakkında bilgi verebilen bir sohbet asistanı sunmayı amaçlar. Proje LangChain-benzeri bileşenler, bir workflow/graph motoru (`langgraph`) ve bir küçük web arayüzü (`streamlit`) kullanır.

## Projenin amacı

- Kullanıcının doğal dilde sorduğu film sorularına (ör. "Bu film nasıl?", "Benim için dram önerir misin?", "Bu filme benzer hangi filmler var?") kısa ve bağlama uygun yanıtlar vermek.
- Çok sayıda film açıklaması ve kullanıcı yorumu içeren dokümanlardan anlamlı bağlam çıkarıp, aranabilir vektör veritabanı (Chroma) üzerinden ilgili parçaları getirmek.
- LLM (OpenRouter/OpenAI uyumlu adapter) ile getirilmiş bağlamı kullanarak doğru ve kibar yanıtlar üretmek.

<img width="863" height="575" alt="image" src="https://github.com/user-attachments/assets/5f644e6d-4062-4f88-b413-954c41185064" />
<img width="1018" height="566" alt="image" src="https://github.com/user-attachments/assets/696d7d45-a419-49c9-96e7-66c21472b947" />
<img width="863" height="575" alt="image" src="https://github.com/user-attachments/assets/fcc94d92-a2e4-4259-923a-1cf21bfb10df" />

## Veri seti hakkında

- Projede `all_movies_reviews.json` adlı bir veri seti kullanılmıştır. Bu dosya film adları, açıklamalar (desc), türler, yönetmenler, kullanıcı yorumları (reviews) ve URL gibi meta verileri içerir.

## Kullanılan yöntemler

- Veri hazırlama ve parçalama: `langchain_text_splitters.RecursiveCharacterTextSplitter` ile metinler uygun büyüklükte chunk'lara bölünür.
- Embedding oluşturma: `langchain_huggingface.HuggingFaceEmbeddings` kullanılarak (ör. Türkçe için önceden eğitilmiş bir BERT modeli) her chunk için embedding üretilir.
- Vektör veritabanı: Chroma (`langchain_community.vectorstores.Chroma`) kullanılarak embedding'ler saklanır ve tekrar sorgulanabilir bir retriever oluşturulur.
- Sorgu yönlendirme (routing): Gelen kullanıcı mesajı önce bir intent sınıflandırıcıdan geçirilir (question_router). Bu intent'e göre akış: film-sorgu ise önce veri getir, sonra LLM ile özet oluştur; genel sohbet ise doğrudan LLM ile cevap üret.
- LLM adapter: `open_router.ChatOpenRouter` (projedeki `open_router.py`) sınıfı ile OpenRouter/OpenAI uyumlu model çağrısı yapılır.
- Workflow/graph: Akış yönetimi için `langgraph` kütüphanesindeki `StateGraph` kullanılır; bu, düğümlere (nodes) çağrılar yaparak state geçişlerini yönetir.

## Elde edilen sonuçlar (özet)

- Film vektör veritabanı oluşturuldu ve lokal disk üzerinde persist edildi (örn. `./.chroma/movie`).
- Kullanıcı sorgularına cevap verirken sistem bağlamdan ilgili pasajları başarılı şekilde getiriyor ve LLM ile bağlama uygun cevaplar oluşturuyor.
- Basit kullanım ile öneri kalitesi model ve embedding seçimlerine bağlıdır; HuggingFace tabanlı Türkçe embeddingler, Türkçe içeriklerde makul performans sağlar.

## Hızlı başlangıç (yerel)

1. Python 3.11 veya uyumlu bir sürüm kurun.
2. Sanal bir ortam oluşturun (önerilir):

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

3. Gerekli paketleri kurun:

   python -m pip install -r requirements.txt

4. Veri setinizi proje klasörüne `all_movies_reviews.json` adıyla koyun.
5. Vektör veritabanını oluşturmak ve veri ingest etmek için:

   python ingestion.py

6. Uygulamayı çalıştırmak için:

   streamlit run main.py

## Web linki

https://your-deployed-app.example.com
