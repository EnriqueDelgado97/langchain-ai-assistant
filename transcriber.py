from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParserLocal


if __name__ == '__main__':
    
    path='temp/'
    glob_pattern='*'
    
    parser = OpenAIWhisperParserLocal(
        device="cpu",  # CPU se usa automáticamente
        lang_model="openai/whisper-small",  # Modelo ligero para CPU
        batch_size=4,  # Ajustar para hardware modesto
        chunk_length=30  # Procesar en fragmentos de 30 segundos
    )
    
    
    loader = GenericLoader.from_filesystem(
        path=path,
        glob=glob_pattern,
        suffixes=[".mp3",".wav",".ogg"],
        show_progress=True,
        parser=parser
    )
    docs = loader.load()
    print(docs)
