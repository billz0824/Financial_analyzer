from data_retrieval.main import retrieve_company_documents

if __name__ == "__main__":
    result = retrieve_company_documents("NVIDIA")
    print("Done! Retrieved files:")
    print(result)



