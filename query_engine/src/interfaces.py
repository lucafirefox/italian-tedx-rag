from typing import Any

import gradio as gr
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer


class RAGQueryInterface:
    """
    Interface for handling RAG (Retrieval-Augmented Generation) queries and chat interactions.

    This class manages the chat engine initialization, query processing, and chat history
    for a RAG-based conversational system.
    """

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.chat_engine = index.as_chat_engine(
            chat_mode="best",
            verbose=True,
            llm=Settings.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=1500),
            system_prompt=(
                "Sei un assistente AI utile e informativo che è stato addestrato "
                "per utilizzare una base di conoscenza. Sii veritiero e non inventare "
                "informazioni. Rispondi alle domande solo utilizzando il contesto fornito."
                "Non aggiungere informazioni non pertinenti."
            ),
        )

    def query(self, query_text: str, chat_history: list) -> tuple[list, str]:
        """
        Process a query and update chat history.

        Args:
            query_text (str): The query text
            chat_history (list): Current chat history

        Returns:
            tuple[list, str]: Updated chat history and empty string for clearing input
        """
        if not query_text.strip():  # Check for empty or whitespace-only input
            return chat_history, ""

        try:
            # Get response from chat engine
            response = self.chat_engine.chat(query_text)

            # Update chat history with new query-response pair
            chat_history.append((query_text, str(response)))

        except Exception as e:
            error_message = "Mi dispiace, si è verificato un errore inaspettato. " f"Dettagli: {e!s}"
            chat_history.append((query_text, error_message))

        # Return updated history and empty string to clear input
        return chat_history, ""

    def reset_chat(self) -> tuple[list[Any], str]:
        """Reset the chat session to its initial state."""
        self.chat_engine.reset()
        return [], ""


class GradioInterface:
    """
    Handles the Gradio interface setup and configuration.

    This class manages the creation and configuration of the Gradio web interface
    components and their associated event handlers.

    Attributes:
        rag_interface (RAGQueryInterface): The RAG query interface instance
        gradio_instance (gr.Blocks): The Gradio interface instance
    """

    def __init__(self, rag_interface: RAGQueryInterface):
        self.rag_interface = rag_interface
        self.gradio_instance = self._create_interface()

    def _create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface components."""
        with gr.Blocks() as gradio_instance:
            gr.Markdown("# Interfaccia RAG - TEDx Italian Talks")

            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
            )

            with gr.Row():
                message_input = gr.Textbox(
                    label="La tua Domanda",
                    placeholder="Digita qui la tua domanda...",
                    lines=2,
                    show_label=True,
                )

            with gr.Row():
                submit_btn = gr.Button("Invia")
                reset_btn = gr.Button("Resetta")

            self._setup_event_handlers(
                chatbot=chatbot, message_input=message_input, submit_btn=submit_btn, reset_btn=reset_btn
            )

        return gradio_instance

    def _setup_event_handlers(
        self, chatbot: gr.Chatbot, message_input: gr.Textbox, submit_btn: gr.Button, reset_btn: gr.Button
    ) -> None:
        """Configure event handlers for interface components."""
        submit_btn.click(
            fn=self.rag_interface.query,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input],
        )

        message_input.submit(
            fn=self.rag_interface.query,
            inputs=[message_input, chatbot],
            outputs=[chatbot, message_input],
        )

        reset_btn.click(
            fn=self.rag_interface.reset_chat,
            inputs=[],
            outputs=[chatbot, message_input],
        )

    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        self.gradio_instance.launch(**kwargs)
