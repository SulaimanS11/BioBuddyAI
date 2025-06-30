# classifier.py

def classify_snake_image(frame):
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from your_vector_loading_script import load_snake_features, load_human_features
    from your_model_setup import model, transform  # wherever your model/transform is

    frame_tensor = transform(frame).unsqueeze(0)
    frame_vec = model.encode_image(frame_tensor).detach().numpy()

    snake_features = load_snake_features()
    human_features = load_human_features()

    best_snake_sim = max([cosine_similarity(frame_vec, vec) for vec in snake_features])
    sim_human = max([cosine_similarity(frame_vec, vec) for vec in human_features])
    adjusted_score = best_snake_sim - sim_human

    best_snake_species = "Boa constrictor"  # placeholder — make this dynamic if needed
    label = f"{best_snake_species} Snake" if adjusted_score > 0.1 else "Human"

    return label, float(adjusted_score)
