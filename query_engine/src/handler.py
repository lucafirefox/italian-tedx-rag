import click
from main import main


@click.command()
@click.option("--chroma-db-path", required=True, type=str, help="Path to store/load the Chroma database")
@click.option("--collection-name", required=True, type=str, help="Name of the Chroma collection")
@click.option("--embedding-model", required=True, type=str, help="Name or path of the embedding model to use")
@click.option("--llm-model", required=True, type=str, help="Name or path of the LLM model to use")
def run(chroma_db_path: str, collection_name: str, embedding_model: str, llm_model: str):
    """
    Handler function to process transcripts and interact with a language model.
    """
    main(
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )


if __name__ == "__main__":
    run()
