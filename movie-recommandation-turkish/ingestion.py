# Os kütüphanesini içe aktarıyoruz.
# Os kütüphanesi, dosya ve dizin işlemleri için gereklidir.
import os
# JSON verilerini okumak için json kütüphanesini içe aktarıyoruz.
import json
# LangChain bileşenlerini içe aktarıyoruz.
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# İlerleme çubuğu için tqdm kütüphanesini kullanıyoruz.
from tqdm import tqdm

# Belgeleri parçalara ayırmak ve vektörleştirmek için gerekli bileşenleri ayarlıyoruz.
# - RecursiveCharacterTextSplitter: metinleri küçük parçalara (chunk) bölerek
#   embedding oluştururken daha iyi sonuç alınmasını sağlar.
#   chunk_size: her parça için hedef boyut (token bazlı tahmini).
#   chunk_overlap: ardışık parçalar arasında örtüşme miktarı, bağlam kaybını azaltır.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=150
)

# Türkçe için eğitilmiş bir BERT tabanlı embedding modeli kullanıyoruz.
# model_kwargs ile cihaz belirleniyor (ör. 'cpu' veya 'cuda').
# encode_kwargs ile embeddinglerin normalize edilmesini isteyebiliriz.
embedding = HuggingFaceEmbeddings(
    model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Chroma veritabanının kaydedileceği dizin
db_path = "./.chroma/movie"

# Eğer veritabanı dizini yoksa veya boşsa, veritabanını oluşturuyoruz.
if not os.path.exists(db_path) or not os.listdir(db_path):
    # Tüm film verilerini JSON'dan okuyoruz (UTF-8 encoding ile)
    with open("all_movies_reviews.json", "r", encoding="utf-8") as f:
        movies_data = json.load(f)

    movie_docs = []
    # Her film için açıklama (desc) ve kullanıcı yorumlarını ayrı Document'lar olarak hazırlıyoruz.
    for movie in movies_data:
        # Metadata'da filmin adı, türü, yönetmeni, puanı ve url'si gibi alanları saklıyoruz.
        meta = {
            "name": movie.get("name"),
            "genre": ", ".join(movie.get("genre", [])),
            "directors": movie.get("directors"),
            "rating": movie.get("rating", {}).get("totalRating"),
            "url": movie.get("url")
        }

        # Filmin genel açıklaması varsa, Document olarak ekle
        if movie.get("desc"):
            movie_docs.append(Document(
                page_content=movie["desc"],
                metadata={**meta, "type": "desc"}
            ))

        # Her bir kullanıcı yorumunu da ayrı bir Document yapıyoruz.
        # metadata içinde yorumun tipini ve kullanıcı puanını ekliyoruz.
        for rev in movie.get("reviews", []):
            if rev.get("review"):
                movie_docs.append(Document(
                    page_content=rev["review"],
                    metadata={**meta, "type": "review", "user_rating": rev.get("rating")}
                ))

    # Oluşturduğumuz Document listelerini belirtilen splitter ile parçalara ayırıyoruz.
    splits = text_splitter.split_documents(movie_docs)

    # Chroma vektör veritabanını başlatıyoruz (daha sonra persist_directory içine kaydedilecek)
    db = Chroma(
        collection_name="movie-db",
        embedding_function=embedding,
        persist_directory=db_path
    )

    # Çok büyük koleksiyonlarda bellek/süre yönetimi için batch ile ekleme yapıyoruz.
    # batch_size'a dikkat: çok büyükse bellek tüketimi artar, çok küçükse yavaş olur.
    batch_size = 5000
    for i in tqdm(range(0, len(splits), batch_size), desc="🔹 Belgeler ekleniyor"):
        batch = splits[i:i + batch_size]
        db.add_documents(batch)

    # İndekslenmiş veritabanını diske kaydediyoruz.
    db.persist()
    print("🎉 Film vektör veritabanı oluşturuldu ve kaydedildi.")
else:
    # Eğer daha önce oluşturulmuşsa yeniden oluşturmayız — bu, geliştirme sürecinde zaman kazandırır.
    print("✅ Film veritabanı zaten mevcut, yeniden oluşturulmadı.")

# Projeyi çalıştırırken kolayca kullanabilmek için bir retriever örneği oluşturuyoruz.
# as_retriever ile döndürdüğümüz nesne aramalarda (k-NN benzeri) kullanılacak.
# search_type: 'similarity' (kullanılan embeddding mesafe metriğine göre benzerlik arar)
# search_kwargs: k = döndürülecek benzer parça sayısı
movie_retriever = Chroma(
    collection_name="movie-db",
    persist_directory=db_path,
    embedding_function=embedding
).as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)
