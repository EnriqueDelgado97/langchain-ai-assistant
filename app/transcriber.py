from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from transformers import WhisperProcessor
from io import BytesIO
import soundfile as sf

class AudioTranscriber:

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="spanish", task="transcribe")
        return OpenAIWhisperParserLocal(
            device="cpu",  # CPU se usa automaticamente
            lang_model="openai/whisper-small",  # Modelo ligero para CPU
            batch_size=4,  # Ajustar para hardware modesto
            chunk_length=30,  # Procesar en fragmentos de 30 segundos
            forced_decoder_ids=forced_decoder_ids
        )

    def transcribe_audio(self, audio_bytes):
        """
        Procesa un archivo de audio en formato bytes y realiza la transcripcion.
        """
        # Guardar los bytes en memoria como un archivo de audio compatible
        with BytesIO(audio_bytes) as audio_file:
            audio_data, sample_rate = sf.read(audio_file)

        # Llamar al parser directamente con los datos de audio
        transcription_result = self.parser.pipe(audio_data, batch_size=4)["text"]

        return transcription_result

if __name__ == '__main__':
    # Simulacion: Lee un archivo de audio como bytes
    with open("temp/tmpnlomzk0e.wav", "rb") as f:  # Cambia por un archivo real
        audio_bytes = f.read()

    # Procesar el audio con AudioTranscriber
    transcriber = AudioTranscriber()
    transcription = transcriber.transcribe_audio(audio_bytes)

    print("Transcripcion:", transcription)