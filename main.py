import argparse
from datetime import datetime

from webpage_embeddings import WebpageEmbeddingGenerator


def run_pipeline(
	urls: list[str],
	bucket_names: list[str],
	region: str,
	chunk_size: int,
	chunk_overlap: int,
	dimension: int,
	output_file: str,
	s3_key: str | None,
) -> None:
	generator = WebpageEmbeddingGenerator(region_name=region)
	all_results = []

	for url in urls:
		try:
			results = generator.process_webpage(
				url=url,
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap,
				dimension=dimension,
			)
			all_results.extend(results)
		except Exception as error:
			print(f"Failed to process {url}: {error}")

	if not all_results:
		raise RuntimeError("No embeddings were generated from the provided URLs.")

	if not bucket_names:
		generator.save_embeddings(results=all_results, output_file=output_file)
		print("Completed: saved embeddings locally (no S3 bucket names provided).")
		return

	for bucket_name in bucket_names:
		upload_key = s3_key
		if not upload_key:
			timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
			upload_key = f"embeddings/{timestamp}-{output_file}"

		generator.save_embeddings(
			results=all_results,
			output_file=output_file,
			bucket_name=bucket_name,
			s3_key=upload_key,
		)
		print(f"Completed: uploaded embeddings to s3://{bucket_name}/{upload_key}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Generate webpage embeddings and optionally upload to S3 buckets."
	)
	parser.add_argument(
		"--urls",
		nargs="+",
		required=True,
		help="One or more webpage URLs to process.",
	)
	parser.add_argument("--s3-buckets", nargs="*", default=[], help="Optional one or more S3 bucket names.")
	parser.add_argument("--region", default="us-east-1", help="AWS region for Bedrock.")
	parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting content.")
	parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks.")
	parser.add_argument("--dimension", type=int, default=1024, help="Embedding dimension.")
	parser.add_argument("--output-file", default="webpage_embeddings.json", help="Output JSON filename.")
	parser.add_argument(
		"--s3-key",
		default=None,
		help="Optional S3 object key for uploaded embeddings. If omitted, a timestamped key is generated.",
	)

	args = parser.parse_args()

	run_pipeline(
		urls=args.urls,
		bucket_names=args.s3_buckets,
		region=args.region,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
		dimension=args.dimension,
		output_file=args.output_file,
		s3_key=args.s3_key,
	)


if __name__ == "__main__":
	main()
