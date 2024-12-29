from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal
from transformers import WhisperProcessor

class AudioTranscriber:

    def __init__(self,path,glob_pattern):
        self.parser = self._create_parser()
        self.loader = self._create_loader(path,glob_pattern)
    def _create_parser(self):
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="spanish", task="transcribe")
        return OpenAIWhisperParserLocal(
            device="cpu",  # CPU se usa automáticamente
            lang_model="openai/whisper-small",  # Modelo ligero para CPU
            batch_size=4,  # Ajustar para hardware modesto
            chunk_length=30,# Procesar en fragmentos de 30 segundos
            forced_decoder_ids=forced_decoder_ids
        )
    
    def _create_loader(self, path, glob_pattern):
        return GenericLoader.from_filesystem(
            path=path,
            glob=glob_pattern,
            suffixes=[".mp3",".wav",".ogg"],
            show_progress=True,
            parser=self.parser
    )

    def run_transcriber(self):
        
        return self.loader.load()



if __name__ == '__main__':
    
    path='temp/'
    glob_pattern='*'
    transcription = AudioTranscriber(path='temp/',glob_pattern='*').run_transcriber()
    print(transcription)