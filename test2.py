from vector_search_and_reranking import vector_search

# results = vector_search(query="dresses under 100", top_k=5, filters={'price': {'$lte': 100}})
results = vector_search(query="dresses under 100", top_k=5, filters={'$or': [{'category': 'dresses'}, {'category': 'tops / blouses'}]})
print(results)
# results = vector_search(query="summer vibe tops", top_k=5, filters=None)
# print(results)
# '$and': [{'category': 'dress'}, {'price': {'$lte': 100}}]}} 


results = vector_search(query="dresses under 100", top_k=6, filters={
  "$and": [
    {"category": "top / blouses"},
    {"price": {"$lte": 100}}
  ]
}
)
print(results)

# results = vector_search(query="dresses under 100", top_k=8,  filters = {"price": {"$lte": 100}})
# print(results)



# filters = {
#     "$and": [
#         {"category": {"$eq": "dress"}},
#         {"price": {"$lte": 150}}
#     ]
# }
# results = vector_search(query="dresses under 100", top_k=6, filters=filters)
# print(results)




# import chromadb

# client = chromadb.PersistentClient(path="./chroma_db")


# collection = client.get_collection(name="products")
# count = collection.count()
# print(f"Collection contains {count} items")

# # Retrieve a sample of items to inspect metadata
# sample = collection.get(include=["metadatas", "documents"], limit=10)
# print("Sample items:")
# for meta, doc in zip(sample["metadatas"], sample["documents"]):
#     print(f"Metadata: {meta}, Document: {doc[:100]}...")