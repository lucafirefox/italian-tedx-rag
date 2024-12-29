import click
from main import main


@click.command()
@click.option("--transcripts-input-dir", required=True, type=str, help="Directory containing transcript files")
@click.option("--chroma-db-path", required=True, type=str, help="Path to store/load the Chroma database")
@click.option("--collection-name", required=True, type=str, help="Name of the Chroma collection")
@click.option("--embedding-model", required=True, type=str, help="Name or path of the embedding model to use")
@click.option("--llm-model", required=True, type=str, help="Name or path of the LLM model to use")
@click.option("--chunk-size", required=True, type=int, help="Refers to how big those chunks of text are")
@click.option("--chunk-overlap", required=True, type=int, help="Refers to how much overlap there is between chunks")
def run(
    transcripts_input_dir: str,
    chroma_db_path: str,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Handler function to process transcripts and interact with a language model.
    """

    main(
        transcripts_input_dir=transcripts_input_dir,
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    run()
