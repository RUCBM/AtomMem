from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss
from openai import OpenAI
import numpy as np

documents = [
    Document(page_content="foo", metadata={"id": 1}),
    Document(page_content="bar", metadata={"id": 2}),
    Document(page_content="baz", metadata={"id": 3}),
    Document(page_content="qux", metadata={"id": 4}),
    Document(page_content="quux", metadata={"id": 5}),
    Document(page_content="corge", metadata={"id": 6}),
    Document(page_content="grault", metadata={"id": 7}),
    Document(page_content="garply", metadata={"id": 8}),
    Document(page_content="waldo", metadata={"id": 9}),
    Document(page_content="fred", metadata={"id": 10}),
    Document(page_content="plugh", metadata={"id": 11}),
    Document(page_content="xyzzy", metadata={"id": 12}),
    Document(page_content="thud", metadata={"id": 13}),
    Document(page_content="fool", metadata={"id": 14}),
    Document(page_content="fooool", metadata={"id": 15}),
    Document(page_content="foolll", metadata={"id": 16}),
    Document(page_content="barrr", metadata={"id": 16}),
    Document(page_content="barrrr", metadata={"id": 17}),
    Document(page_content="barrrrr", metadata={"id": 18}),
    Document(page_content="barrrrrr", metadata={"id": 19}),
    Document(page_content="barrrrrrr", metadata={"id": 20}),
    Document(page_content="fscas", metadata={"id": 21}),
    Document(page_content="fscasas", metadata={"id": 22}),
    Document(page_content="fscasasas", metadata={"id": 23}),
    # Document(page_content="fscasasasas", metadata={"id": 24}),
]

def call_remote_embedding(texts):
        base_url = 'http://localhost:8007/v1/'
        model_name = 'qwen3-embedding'
        client = OpenAI(api_key='sk-123', base_url=base_url)
        response = client.embeddings.create(
            input=texts,
            model=model_name,
        ).model_dump()
        if len(response['data']) > 1:
            embeddings = [item['embedding'] for item in response['data']]
        else:
            embeddings = response['data'][0]['embedding']
        np_embeddings = np.array(embeddings, dtype=np.float32)
        return np_embeddings


faiss_index = faiss.IndexFlatL2(1024)
vectorstore = FAISS(call_remote_embedding, faiss_index, docstore= InMemoryDocstore(),
                index_to_docstore_id={})

vectorstore.add_documents(documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": {"id": 1}, "fetch_k": 50})
memory = retriever.invoke('1234')

print(memory)
