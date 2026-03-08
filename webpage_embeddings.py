import boto3
import json
import os
from datetime import datetime
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utilities.upload_s3_file import upload_file_to_s3


class WebpageEmbeddingGenerator:
    def __init__(self, region_name='us-east-1'):
        """Initialize AWS Bedrock client for Nova embeddings."""
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = 'amazon.nova-2-multimodal-embeddings-v1:0'
    
    def fetch_webpage_content(self, url):
        """
        Fetch and parse content from a live webpage.
        
        Args:
            url: URL of the webpage to parse
            
        Returns:
            List of document objects with content and metadata
        """
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            print(f"✓ Successfully fetched content from: {url}")
            return documents
        except Exception as e:
            print(f"✗ Error fetching webpage: {e}")
            raise
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of documents to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        return chunks
    
    def generate_embedding(self, text, dimension=1024):
        """
        Generate embedding for text using AWS Nova.
        
        Args:
            text: Text to embed
            dimension: Embedding dimension 
            
        Returns:
            Embedding vector
        """
        try:
            request_body = {
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": 3072,
                    "text": {"truncationMode": "END", "value": text},
                }
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            #print(json.dumps(response_body, indent=2))
            return response_body
            
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            raise
    
    def process_webpage(self, url, chunk_size=1000, chunk_overlap=200, dimension=3074):
        """
        Complete pipeline: fetch webpage, chunk, and create embeddings.
        
        Args:
            url: URL to process
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            dimension: Embedding dimension
            
        Returns:
            List of dictionaries with chunks and their embeddings
        """
        print(f"\n🌐 Processing webpage: {url}")
        print("=" * 60)
        
        # Fetch content
        documents = self.fetch_webpage_content(url)
        
        # Chunk documents
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        
        # Generate embeddings
        results = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...", end='\r')
            
            embedding = self.generate_embedding(chunk.page_content, dimension)
            
            result = {
                'chunk_id': i,
                'text': chunk.page_content,
                'embedding': embedding,
                'metadata': {
                    'source': chunk.metadata.get('source', url),
                    'title': chunk.metadata.get('title', ''),
                    'chunk_size': len(chunk.page_content)
                }
            }
            results.append(result)
        
        print(f"\n✓ Generated {len(results)} embeddings")
        return results
    
    def save_embeddings(self, results, output_file='embeddings_output.json', bucket_name=None, s3_key=None):
        """
        Save embeddings to a JSON file with timestamp and upload to S3.
        
        Args:
            results: List of embedding results
            output_file: Path to output file (timestamp will be added)
            bucket_name: S3 bucket name (optional; defaults to EMBEDDINGS_BUCKET env var)
            s3_key: S3 object key (optional; defaults to output file name)
        """
        try:
            output_dir = 'embedding_library'
            os.makedirs(output_dir, exist_ok=True)

            # Add timestamp to filename
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            base_name = os.path.basename(output_file)
            name, ext = os.path.splitext(base_name)
            timestamped_name = f"{name}_{timestamp}{ext}"
            output_path = os.path.join(output_dir, timestamped_name)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved embeddings to: {output_path}")

            target_bucket = bucket_name or os.getenv('EMBEDDINGS_BUCKET')
            if target_bucket:
                upload_key = s3_key or os.path.basename(output_path)
                s3_uri = upload_file_to_s3(target_bucket, output_path, upload_key)
                print(f"✓ Uploaded embeddings to: {s3_uri}")
            else:
                print("ℹ Skipping S3 upload (set bucket_name or EMBEDDINGS_BUCKET)")
        except Exception as e:
            print(f"✗ Error saving embeddings: {e}")
            raise


def main():
    """Example usage of the WebpageEmbeddingGenerator."""
    
    # Initialize generator
    generator = WebpageEmbeddingGenerator(region_name='us-east-1')
    
    # URLs to process
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        # Add more URLs here
    ]
    
    all_results = []
    
    for url in urls:
        try:
            # Process webpage and generate embeddings
            results = generator.process_webpage(
                url=url,
                chunk_size=1000,
                chunk_overlap=200,
                dimension=1024  # Options: 256, 512, 1024
            )
            
            all_results.extend(results)
            
            # Print sample result
            if results:
                print("\n📄 Sample chunk:")
                print(f"Text: {results[0]['text'][:200]}...")
                print(f"Embedding dimension: {len(results[0]['embedding'])}")
                print(f"Metadata: {results[0]['metadata']}")
            
        except Exception as e:
            print(f"Failed to process {url}: {e}")
            continue
    
    # Save all embeddings
    if all_results:
        generator.save_embeddings(all_results, 'webpage_embeddings.json')
        print(f"\n✅ Total embeddings generated: {len(all_results)}")


if __name__ == "__main__":
    main()
