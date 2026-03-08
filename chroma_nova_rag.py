import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
import chromadb
from chromadb.config import Settings


class ChromaNovaRAG:
    def __init__(
        self,
        region_name: str = "us-east-1",
        embedding_model_id: str = "amazon.nova-2-multimodal-embeddings-v1:0",
        chat_model_id: str = "amazon.nova-lite-v1:0",
        persist_directory: str = "chroma_db",
        collection_name: str = "webpage_embeddings",
    ) -> None:
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)
        self.embedding_model_id = embedding_model_id
        self.chat_model_id = chat_model_id

        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _extract_embedding_vector(embedding_payload: Any) -> List[float]:
        if isinstance(embedding_payload, list):
            return embedding_payload

        if isinstance(embedding_payload, dict):
            direct = embedding_payload.get("embedding")
            if isinstance(direct, list):
                return direct

            embeddings = embedding_payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first_item = embeddings[0]
                if isinstance(first_item, dict) and isinstance(first_item.get("embedding"), list):
                    return first_item["embedding"]

            embeddings_list = embedding_payload.get("embeddingsList")
            if isinstance(embeddings_list, list) and embeddings_list:
                first_item = embeddings_list[0]
                if isinstance(first_item, list):
                    return first_item
                if isinstance(first_item, dict) and isinstance(first_item.get("embedding"), list):
                    return first_item["embedding"]

        raise ValueError("Unsupported embedding payload format")

    def embed_text(self, text: str, dimension: int = 1024) -> List[float]:
        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": dimension,
                "text": {"truncationMode": "END", "value": text},
            },
        }

        response = self.bedrock_runtime.invoke_model(
            modelId=self.embedding_model_id,
            body=json.dumps(request_body),
        )
        response_body = json.loads(response["body"].read())
        return self._extract_embedding_vector(response_body)

    def build_index_from_embeddings_file(self, embeddings_file: str) -> int:
        if not os.path.isfile(embeddings_file):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        with open(embeddings_file, "r", encoding="utf-8") as file_handle:
            records = json.load(file_handle)

        ids: List[str] = []
        vectors: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for record in records:
            chunk_id = record.get("chunk_id")
            text = record.get("text", "")
            metadata = record.get("metadata", {}) or {}
            vector = self._extract_embedding_vector(record.get("embedding"))

            source = metadata.get("source", "")
            title = metadata.get("title", "")
            doc_id = f"{source}::{chunk_id}" if source else f"chunk::{chunk_id}"

            ids.append(doc_id)
            vectors.append(vector)
            documents.append(text)
            metadatas.append(
                {
                    "source": str(source),
                    "title": str(title),
                    "chunk_id": int(chunk_id) if isinstance(chunk_id, int) else -1,
                }
            )

        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def retrieve(self, question: str, top_k: int = 5, embedding_dimension: int = 1024) -> Dict[str, Any]:
        question_embedding = self.embed_text(question, embedding_dimension)

        query_result = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return query_result

    @staticmethod
    def _build_context_from_results(query_result: Dict[str, Any]) -> Tuple[str, List[str]]:
        documents = (query_result.get("documents") or [[]])[0]
        metadatas = (query_result.get("metadatas") or [[]])[0]

        context_blocks: List[str] = []
        sources: List[str] = []

        for index, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
            source = (metadata or {}).get("source", "unknown")
            title = (metadata or {}).get("title", "")
            chunk_id = (metadata or {}).get("chunk_id", "")
            sources.append(str(source))

            context_blocks.append(
                f"Source {index}: {source}\n"
                f"Title: {title}\n"
                f"Chunk: {chunk_id}\n"
                f"Content:\n{document}"
            )

        return "\n\n".join(context_blocks), sorted(set(sources))

    def generate_answer(self, question: str, query_result: Dict[str, Any]) -> Dict[str, Any]:
        context_text, sources = self._build_context_from_results(query_result)

        system_prompt = (
            "You are a RAG assistant. Answer only from provided context. "
            "If context is insufficient, say you do not have enough information. "
            "Always include a short Sources section with the exact URLs used."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            "Provide a concise answer and then list Sources."
        )

        response = self.bedrock_runtime.converse(
            modelId=self.chat_model_id,
            system=[{"text": system_prompt}],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_prompt}],
                }
            ],
            inferenceConfig={"maxTokens": 700, "temperature": 0.2},
        )

        content = response.get("output", {}).get("message", {}).get("content", [])
        answer_text = ""
        if content and isinstance(content[0], dict):
            answer_text = content[0].get("text", "")

        return {
            "answer": answer_text,
            "sources": sources,
            "retrieval": query_result,
        }



def main() -> None:
    parser = argparse.ArgumentParser(description="Chroma + Nova RAG pipeline")
    parser.add_argument(
        "--embeddings-file",
        default="embedding_library/webpage_embeddings.json",
        help="Path to embeddings JSON created by webpage_embeddings.py",
    )
    parser.add_argument("--collection", default="webpage_embeddings", help="Chroma collection name")
    parser.add_argument("--persist-dir", default="chroma_db", help="Chroma persist directory")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks for retrieval")
    parser.add_argument("--question", required=True, help="User question for RAG query")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild index from embeddings file before querying",
    )

    args = parser.parse_args()

    rag = ChromaNovaRAG(
        region_name=args.region,
        persist_directory=args.persist_dir,
        collection_name=args.collection,
    )

    if args.reindex or rag.collection.count() == 0:
        indexed_count = rag.build_index_from_embeddings_file(args.embeddings_file)
        print(f"Indexed chunks: {indexed_count}")
    else:
        print(f"Using existing index with {rag.collection.count()} chunks")

    retrieval = rag.retrieve(args.question, top_k=args.top_k)
    result = rag.generate_answer(args.question, retrieval)

    print("\nAnswer:\n")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
