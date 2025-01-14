from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import ScoredPoint
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import arxiv_tool
import json

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
        
        try:
            for paper in papers:
                if float(paper.score) < self.threshold:
                    break # we assume that papers are sorted by similarity

                data = arxiv_tool.get_paper(paper.payload['arxiv-id'])

                print('Selected', paper.payload['title'], paper.payload['arxiv-id'])

                #content = '<paper><title>' + paper.payload['title'] + '</title><content>' + data['text'] + '</content></paper>'
                content = dict()
                content['title'] = paper.payload['title']
                content['authors'] = paper.payload['authors']
                content['arxiv-id'] = paper.payload['arxiv-id']
                content['link'] = data['arxiv_link']
                content['published'] = paper.payload['published']
                content['text'] = data['text']

                #results.append(Document(page_content=content, metadata={k:v for k,v in data.items() if k != 'text'}))
                results.append(Document(page_content=json.dumps(content)))

                ## Should we check for tokens left?

            return results
        except Exception as e:
            print(e)
            exit()