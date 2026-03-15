import os
import json
import boto3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from chroma_nova_rag import ChromaNovaRAG
from webpage_embeddings import WebpageEmbeddingGenerator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval: Dict[str, Any]


class ReindexRequest(BaseModel):
    embeddings_file: Optional[str] = Field(
        default=None,
        description="Path to embeddings JSON file",
        example="embedding_library/webpage_embeddings.json",
    )


class IngestMultipleURLsRequest(BaseModel):
    topic_name: str = Field(..., min_length=1, description="Topic name used to label and name the embeddings file")
    urls: List[str] = Field(..., min_items=1, description="One or more webpage URLs to ingest")
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    dimension: Literal[256, 384, 1024, 3072] = Field(default=3072, description="Embedding dimension")


class IngestMultipleURLsResponse(BaseModel):
    status: str
    total_chunks_generated: int
    urls_processed: List[str]
    urls_failed: List[str]
    embeddings_file: str
    message: str


class EmbeddingsFileMetadata(BaseModel):
    filename: str
    size_bytes: int
    created_timestamp: str
    path: str
    source_urls: List[str]
    source_url_count: int


class ListEmbeddingsFilesResponse(BaseModel):
    total_files: int
    embeddings_directory: str
    files: List[EmbeddingsFileMetadata]


class S3EmbeddingsFileMetadata(BaseModel):
    s3_key: str
    size_bytes: int
    last_modified: str
    source_urls: List[str]
    source_url_count: int


class ListS3EmbeddingsResponse(BaseModel):
    bucket: str
    prefix: str
    total_files: int
    files_processed: int
    files: List[S3EmbeddingsFileMetadata]


APP_REGION = os.getenv("AWS_REGION", "us-east-1")
APP_COLLECTION = os.getenv("RAG_COLLECTION", "webpage_embeddings")
APP_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "chroma_db")
APP_EMBEDDINGS_FILE = os.getenv("RAG_EMBEDDINGS_FILE", "embedding_library/webpage_embeddings.json")

app = FastAPI(
    title="Votan Nova RAG API",
    version="1.0.0",
    description="RAG pipeline with Chroma vector index and AWS Nova embeddings/chat models",
)
rag = ChromaNovaRAG(
    region_name=APP_REGION,
    persist_directory=APP_PERSIST_DIR,
    collection_name=APP_COLLECTION,
)
webpage_generator = WebpageEmbeddingGenerator(region_name=APP_REGION)


@app.on_event("startup")
def startup_index() -> None:
    try:
        if rag.collection.count() == 0 and os.path.isfile(APP_EMBEDDINGS_FILE):
            rag.build_index_from_embeddings_file(APP_EMBEDDINGS_FILE)
    except Exception as error:
        print(f"Startup indexing skipped: {error}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "collection": APP_COLLECTION,
        "persist_dir": APP_PERSIST_DIR,
        "chunk_count": rag.collection.count(),
    }


@app.get("/embeddings", response_model=ListEmbeddingsFilesResponse, summary="List Local Embeddings Files")
def list_embeddings_files() -> ListEmbeddingsFilesResponse:
    try:
        embeddings_dir = Path("embedding_library")
        if not embeddings_dir.exists():
            return ListEmbeddingsFilesResponse(total_files=0, embeddings_directory=str(embeddings_dir), files=[])

        files_metadata = []
        for json_file in sorted(embeddings_dir.glob("*.json")):
            stat = json_file.stat()
            created_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
            source_urls = set()

            try:
                with open(json_file, "r", encoding="utf-8") as file_handle:
                    records = json.load(file_handle)

                if isinstance(records, list):
                    for record in records:
                        metadata = record.get("metadata", {}) or {}
                        source = metadata.get("source") or metadata.get("source_url")
                        if source:
                            source_urls.add(str(source))
            except Exception:
                source_urls = set()

            files_metadata.append(
                EmbeddingsFileMetadata(
                    filename=json_file.name,
                    size_bytes=stat.st_size,
                    created_timestamp=created_time,
                    path=str(json_file),
                    source_urls=sorted(source_urls),
                    source_url_count=len(source_urls),
                )
            )

        return ListEmbeddingsFilesResponse(
            total_files=len(files_metadata),
            embeddings_directory=str(embeddings_dir),
            files=files_metadata,
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/s3-embeddings", response_model=ListS3EmbeddingsResponse, summary="List S3 Embeddings Files with URLs")
def list_s3_embeddings_files(
    bucket: str = Query(..., description="S3 bucket name"),
    prefix: str = Query("", description="Optional prefix to filter files"),
    region: str = Query("us-east-1", description="AWS region"),
) -> ListS3EmbeddingsResponse:
    try:
        s3_client = boto3.client("s3", region_name=region)
        files_metadata = []
        files_processed = 0

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                s3_key = obj["Key"]
                if not s3_key.endswith(".json"):
                    continue

                source_urls = set()

                try:
                    response = s3_client.get_object(Bucket=bucket, Key=s3_key)
                    content = response["Body"].read().decode("utf-8")
                    records = json.loads(content)

                    files_processed += 1

                    if isinstance(records, list):
                        for record in records:
                            metadata = record.get("metadata", {}) or {}
                            source = metadata.get("source") or metadata.get("source_url")
                            if source:
                                source_urls.add(str(source))

                    last_modified = obj["LastModified"].isoformat()
                    files_metadata.append(
                        S3EmbeddingsFileMetadata(
                            s3_key=s3_key,
                            size_bytes=obj["Size"],
                            last_modified=last_modified,
                            source_urls=sorted(source_urls),
                            source_url_count=len(source_urls),
                        )
                    )

                except Exception as parse_error:
                    print(f"Warning: Could not parse {s3_key}: {parse_error}")

        return ListS3EmbeddingsResponse(
            bucket=bucket,
            prefix=prefix,
            total_files=len(files_metadata),
            files_processed=files_processed,
            files=files_metadata,
        )

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/reindex")
def reindex(payload: ReindexRequest) -> Dict[str, Any]:
    try:
        embeddings_file = payload.embeddings_file or APP_EMBEDDINGS_FILE
        indexed_count = rag.build_index_from_embeddings_file(embeddings_file)
        return {
            "status": "indexed",
            "embeddings_file": embeddings_file,
            "indexed_count": indexed_count,
            "collection_count": rag.collection.count(),
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    try:
        if rag.collection.count() == 0:
            if os.path.isfile(APP_EMBEDDINGS_FILE):
                rag.build_index_from_embeddings_file(APP_EMBEDDINGS_FILE)
            else:
                raise ValueError(
                    "No index available. Run /reindex first or set RAG_EMBEDDINGS_FILE to a valid file."
                )

        retrieval = rag.retrieve(payload.question, top_k=payload.top_k)
        result = rag.generate_answer(payload.question, retrieval)

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            retrieval=result.get("retrieval", {}),
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/ingest", response_model=IngestMultipleURLsResponse)
def ingest_multiple_urls(payload: IngestMultipleURLsRequest) -> IngestMultipleURLsResponse:
    try:
        all_results = []
        urls_processed = []
        urls_failed = []

        for url in payload.urls:
            try:
                results = webpage_generator.process_webpage(
                    url=url,
                    chunk_size=payload.chunk_size,
                    chunk_overlap=payload.chunk_overlap,
                    dimension=payload.dimension,
                )
                all_results.extend(results)
                urls_processed.append(url)
            except Exception as url_error:
                urls_failed.append(f"{url}: {str(url_error)}")

        if not all_results:
            raise ValueError("No embeddings were generated from the provided URLs.")

        ingested_at = datetime.utcnow().isoformat() + "Z"
        for record in all_results:
            if isinstance(record.get("metadata"), dict):
                record["metadata"]["ingested_at"] = ingested_at
                record["metadata"]["topic"] = payload.topic_name
            else:
                record["ingested_at"] = ingested_at
                record["topic"] = payload.topic_name

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in payload.topic_name)
        output_file = f"{safe_topic}_{timestamp}.json"
        output_dir = Path("embedding_library")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file

        with open(output_path, "w", encoding="utf-8") as file_handle:
            json.dump(all_results, file_handle, indent=2, ensure_ascii=False)


        return IngestMultipleURLsResponse(
            status="success",
            total_chunks_generated=len(all_results),
            urls_processed=urls_processed,
            urls_failed=urls_failed,
            embeddings_file=str(output_path),
            message=f"Processed {len(urls_processed)} URLs, generated {len(all_results)} chunks.",
        )
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
