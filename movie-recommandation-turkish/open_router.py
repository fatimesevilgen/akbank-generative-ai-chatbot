# Standart kütüphaneden: ortam değişkenlerine erişim için kullanılır.
import os

# typing araçları: Optional bir değerin None olabileceğini belirtmek için kullanılır.
from typing import Optional

# .env dosyasını yüklemek için: geliştirme ortamında gizli anahtarları
# .env dosyasından almak için kullanıyoruz.
from dotenv import load_dotenv

# LangChain helper: environment'dan gizli anahtar almak için küçük bir yardımcı.
from langchain_core.utils.utils import secret_from_env

# LangChain/OpenAI uyumlu Chat modelinin temel sınıfı.
from langchain_openai import ChatOpenAI

# Pydantic alan tipleri: Field ile alias/default_factory tanımlanır,
# SecretStr ise gizli anahtarları güvenli tutmak için kullanılır.
from pydantic import Field, SecretStr


# .env dosyasındaki değişkenleri yükle. Bu, local geliştirme sırasında
# OPENROUTER_API_KEY gibi gizli anahtarların .env dosyasından okunmasını sağlar.
load_dotenv()


class ChatOpenRouter(ChatOpenAI):
    """
    OpenRouter uyumlu küçük bir ChatOpenAI alt sınıfı.

    Amaç:
    - LangChain'in ChatOpenAI sınıfını genişleterek OpenRouter için kolay
    - konfigürasyon (base_url, api_key) desteği sağlamak.

    Özellikler:
    - `openai_api_key` alanı Pydantic Field ile alias (api_key) ve
      default_factory kullanılarak env'den okunur.
    - `lc_secrets` property ile LangChain'e hangi environment
      değişkeninin gizli olduğunu bildiriyoruz.
    """

    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        # LangChain secrets mapping: model tarafında kullanılacak gizli anahtarın
        # hangi env değişkeninden geleceğini bildirir.
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 **kwargs):
        # Eğer __init__'e açıkça api_key verilmediyse, environment değişkeninden al.
        openai_api_key = (
            openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        )
        # ChatOpenAI'in constructor'ına base_url ve api_key vererek OpenRouter'ı hedefliyoruz.
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_api_key,
            **kwargs
        )


# Proje genelinde kullanılacak LLM örneğini oluşturuyoruz.
# model parametresi ve temperature gibi ayarlar burada belirlenir.
llm = ChatOpenRouter(model="openai/gpt-5-nano", temperature=0.7)