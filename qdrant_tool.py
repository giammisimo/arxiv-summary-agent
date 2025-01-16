from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import ScoredPoint
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import arxiv_tool
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class Qdrant_tool(BaseRetriever):
    # Define all fields explicitly - pydantic
    """
    Qdrant retriever class

    Fields:
        `host` (`str`): L'indirizzo dell'istanza Qdrant
        `port` (`int`): La porta dell'istanza Qdrant
        `collection` (`str`): La collezione Qdrant in cui cercare
        `model` (`str`): Il modello da utilizzare per calcolare l'embedding delle query.
        Il modello deve essere compatibile con SentenceTransformer e lo stesso
        utilizzato per i vettori serviti da Qdrant. default = `'all-mpnet-base-v2'`
        `threshold` (`float`): Soglia minima di somiglianza per accettare un risultato della ricerca.
        Default = `0.4`
    """
    host: str
    port: int
    collection: str
    embedding_model: str = "all-mpnet-base-v2"
    threshold: float = 0.4
    top_k: int = 5

    client: QdrantClient = None
    encoder: SentenceTransformer = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the client and embedding model
        self.client = QdrantClient(host=self.host, port=self.port)
        self.encoder = SentenceTransformer(self.embedding_model)

    def embed_and_search(self, collection: str, query: str, top_k: int) -> list[ScoredPoint]:
        """
        Embedda una frase e cerca nel database i documenti più simili.

        Args:
            `query` (`str`): La frase da embeddare e cercare.
            `top_k` (`int`): Il numero di risultati più simili da restituire.

        Returns:
            `List[ScoredPoint]`: Lista di punti più simili.
        """
        print('QDRANT CALLED')
        #client = QdrantClient(host=self.host, port=self.port)
        #embedding_model = SentenceTransformer(self.embedding_model)
        query_embedding = self.encoder.encode(query, convert_to_tensor=False).tolist()

        search_results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k,
        )

        return search_results
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        papers = self.embed_and_search(self.collection, query, self.top_k)

        results: List[Document] = []
        
        with ThreadPoolExecutor() as executor:
            future_to_paper = {executor.submit(self.fetch_paper, paper): paper for paper in papers}
            
            for future in as_completed(future_to_paper):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error in thread: {e}")

        return results
    
    def fetch_paper(self,paper):
        """
        Helper function to fetch a paper's data and structure it.
        """
        try:
            if float(paper.score) < self.threshold:
                return None  # Skip papers below the threshold

            data = arxiv_tool.get_paper(paper.payload['arxiv-id'])
            print('Selected', paper.payload['title'], paper.payload['arxiv-id'])

            content = {
                'title': paper.payload['title'],
                'authors': paper.payload['authors'],
                'arxiv-id': paper.payload['arxiv-id'],
                'link': data['arxiv_link'],
                'published': paper.payload['published'],
                'text': data['text']
            }

            return Document(page_content=json.dumps(content))
        except Exception as e:
            print(f"Error processing paper {paper.payload['arxiv-id']}: {e}")
            return None