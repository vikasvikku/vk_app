import spacy
from typing import List

class TextChunker:
    @staticmethod
    def chunk_text_using_spacy(text: str, max_tokens: int = 512, overlap: int = 10) -> List[str]:
        """Split text into chunks of specified size using SpaCy while preserving sentence boundaries."""
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_chunk = []
        current_token_count = 0

        for sentence in sentences:
            sentence_doc = nlp(sentence)
            sentence_tokens = [token.text for token in sentence_doc]
            num_tokens_in_sentence = len(sentence_tokens)

            if num_tokens_in_sentence > max_tokens:
                # If the sentence itself is longer than max_tokens, split it into chunks
                start = 0
                end = max_tokens
                while start < num_tokens_in_sentence:
                    chunk = " ".join(sentence_tokens[start:end])
                    chunks.append(chunk)
                    start += max_tokens - overlap
                    end = min(start + max_tokens, num_tokens_in_sentence)
                current_chunk = []
                current_token_count = 0
                continue

            if current_token_count + num_tokens_in_sentence > max_tokens:
                # If adding this sentence exceeds max_tokens, finalize the current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0

            current_chunk.append(sentence)
            current_token_count += num_tokens_in_sentence

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
        """Use the SpaCy chunking method instead of the traditional method."""
        return TextChunker.chunk_text_using_spacy(text, max_tokens=chunk_size)