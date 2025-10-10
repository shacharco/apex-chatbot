import pickle

# Load (read) a pickle file
with open("indices/travel/doc_ids.pkl", "rb") as f:  # 'rb' = read binary
    data = pickle.load(f)

print(type(data))
print(data)
