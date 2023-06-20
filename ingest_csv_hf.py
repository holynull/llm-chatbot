"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import argparse

if getattr(sys, 'frozen', False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

parser = argparse.ArgumentParser(description='Ingest data.')
parser.add_argument('-f', '--fileName',
                    help="CSV file name")
parser.add_argument('-n', '--name',
                    help="pkl file name")
args = parser.parse_args()
fileNmae = args.fileName
name = args.name


def ingest_docs():
    """Get documents from web pages."""
    loader = CSVLoader(fileNmae)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings()
    if os.path.exists(f"{script_location}/{name}.pkl"):
        with open(f"{name}.pkl", "rb") as f:
            vectorstore = pickle.load(f)
        n_vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.merge_from(n_vectorstore)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    if fileNmae==None or name==None:
        parser.print_usage()
    else:
        ingest_docs()
