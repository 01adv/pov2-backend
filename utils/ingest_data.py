import pandas as pd
import chromadb
from chromadb.config import Settings
from utils.rag import add_document
chroma_client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(anonymized_telemetry=False)
)


def ingest_data():
    try:
        # Load Excel data from uploaded file path
        df = pd.read_csv("formatted_products.csv")

        # Try to get existing ChromaDB collection
        try:
            collection = chroma_client.get_collection("products")
            return  # Exit if collection already exists
        except Exception:
            collection = chroma_client.create_collection("products")

        # Ingest each row from the DataFrame
        for index, row in df.iterrows():
            metadata = {
                "id": str(index),
                "Product Name": row.get("Product Name"),
                "category": row.get("Category"),
                # "Type": row.get("Type"),
                "Tags": row.get("Tags"),
                "Price": row.get("Price"),
                "Product Description": row.get("Product Description"),
                "SEO Title": row.get("SEO Title"),
                "SEO Description": row.get("SEO Description"),
                "Available Variants": row.get("Available Variants"),
                "Product Details": row.get("Product Details"),
                "Vibe": row.get("Vibe"),
            }

            # Create a descriptive text for embedding
            def format_dict_as_sentence(d):
                """Convert a dictionary to a sentence-like string."""
                parts = []
                for key, value in d.items():
                    if isinstance(value, list):
                        if value:  # Non-empty list
                            parts.append(
                                f"{key.title()} - {', '.join(str(v) for v in value)}")
                    elif pd.notnull(value) and str(value).strip():
                        parts.append(f"{key.title()} - {value}")
                return ". ".join(parts) + "." if parts else ""

            text_parts = []

            for key, value in metadata.items():
                if not pd.notnull(value):
                    continue
                if key == "Tags":
                    print(type(value))
                    if value == "[, ]":
                        continue  # Skip empty Tags list
                    value = ", ".join(value)
                elif key == "Available Variants" and isinstance(value, dict):
                    value = format_dict_as_sentence(value)
                elif key == "Product Details" and isinstance(value, dict):
                    value = format_dict_as_sentence(value)

                if key in [
                    "Product Name", "category", "Product Details", "Product Description", "SEO Title",
                    "SEO Description", "Tags", "Price", "Available Variants", "Vibe"
                ]:
                    text_parts.append(
                        f"{key.replace('_', ' ').title()}: {value}")

            # Resulting formatted string list
            # print(text_parts)

            text = "\n".join(text_parts)
            print(f"Ingesting document: {text}")
            # break

            # Add document to the collection
            add_document(collection, text, metadata)

    except Exception:
        raise
