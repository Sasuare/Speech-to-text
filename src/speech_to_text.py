import whisper
import json
from pathlib import Path
from .normalizer.base import TextNormalizer


# -------------------------
# Mock client (temporal)
# -------------------------
class MockLLMClient:
    """
    Cliente simulado.
    En producción se reemplaza por OpenAI, local LLM, etc.
    """
    def generate(self, prompt: str) -> str:
        # Por ahora devuelve el texto original sin modificar
        # (sirve para validar el pipeline completo)
        return prompt.split("Texto:")[-1].strip()


# -------------------------
# Normalizador contextual
# -------------------------
class LLMNormalizer(TextNormalizer):
    def __init__(self, client):
        self.client = client

    def normalize(self, text: str) -> str:
        prompt = f"""
Normaliza el siguiente texto del español colombiano hablado
a un español neutro, claro y natural.


Texto:
{text}
"""
        response = self.client.generate(prompt)
        return response.strip()


# -------------------------
# Pipeline principal
# -------------------------
def transcribir_y_traducir(
    ruta_audio: Path,
    modelo: str = "base",
    idioma_origen: str = "es"
) -> dict:
    """
    Transcribe audio en español, normaliza el texto
    y traduce a inglés manteniendo timestamps.
    """

    if not ruta_audio.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_audio}")

    print("🔹 Cargando modelo Whisper...")
    model = whisper.load_model(modelo)

    print("🔹 Transcribiendo (ES)...")
    transcripcion = model.transcribe(
        str(ruta_audio),
        language=idioma_origen,
        fp16=False
    )

    print("🔹 Traduciendo a inglés (EN)...")
    traduccion = model.transcribe(
        str(ruta_audio),
        task="translate",
        fp16=False
    )

    # Inicializar normalizador
    normalizer = LLMNormalizer(client=MockLLMClient())

    segmentos_finales = []

    for seg_es, seg_en in zip(transcripcion["segments"], traduccion["segments"]):
        texto_original = seg_es["text"].strip()
        texto_normalizado = normalizer.normalize(texto_original)

        segmentos_finales.append({
            "start": seg_es["start"],
            "end": seg_es["end"],
            "text_es": texto_normalizado,
            "text_en": seg_en["text"].strip()
        })

    return {
        "language_source": "es",
        "language_target": "en",
        "text_es": " ".join(s["text_es"] for s in segmentos_finales),
        "text_en": traduccion["text"],
        "segments": segmentos_finales
    }


# -------------------------
# Persistencia
# -------------------------
def guardar_json(data: dict, ruta_salida: Path):
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Archivo guardado en: {ruta_salida}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    audio_path = Path("data/audio_test.wav")
    output_path = Path("data/output/audio_test_es_en.json")

    resultado = transcribir_y_traducir(audio_path)

    guardar_json(resultado, output_path)

    print("\n📝 TEXTO ES (NORMALIZADO):")
    print(resultado["text_es"])

    print("\n📝 TEXT EN:")
    print(resultado["text_en"])
