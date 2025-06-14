def answer_question(question: str, image_base64: str = None):
    if image_base64:
        image_text = extract_text_from_image(image_base64)
        question += f"\n\nImage content:\n{image_text}"
    
    docs = search_vector_db(question)
    links = extract_relevant_links(docs)

    answer = generate_answer(question, docs)
    
    return {
        "answer": answer,
        "links": links
    }
