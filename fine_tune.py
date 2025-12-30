import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader

def fine_tune_specter(output_path="./models/specter_matchmaker_v1", epochs=1):
    """
    Fine-tune SPECTER to distinguish Methodological matches from Topical matches.
    """
    # 1. Load Base Model
    model_name = 'allenai/specter'
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Define Training Data (Triplets)
    # Anchor: The paper in question
    # Positive: Methodologically similar (Implicit Link)
    # Negative: Topically similar but methodologically distinct (Distractor)
    
    # Synthetic examples for demonstration
    train_examples = [
        InputExample(texts=[
            "Liquid Time-Constant Networks [SEP] Continuous-time neural networks with adaptive time constants.", 
            "C. elegans Gap Junction Dynamics [SEP] Biological neural networks with continuous synaptic transmission.", 
            "History of Convolutional Networks [SEP] Overview of CNNs and deep learning history."
        ]),
        InputExample(texts=[
            "Graph Neural Networks for Molecule Discovery [SEP] Using GNNs to predict molecular properties.",
            "3D Protein folding prediction [SEP] Deep learning for structural biology and drug discovery.",
            "Social Network Analysis [SEP] Analyzing graph structures in social media."
        ]),
         InputExample(texts=[
            "Transformer Interpretability with Attention Rollout [SEP] Visualizing attention flow in transformers.",
            "Causal Inference in Complex Systems [SEP] Understanding cause and effect in high-dimensional data.",
            " BERT for Sentiment Analysis [SEP] classifying text sentiment using transformers."
        ])
    ]

    # 3. Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

    # 4. Define Loss
    # TripletLoss minimizes distance(anchor, positive) while maximizing distance(anchor, negative)
    train_loss = losses.TripletLoss(model=model)

    # 5. Train
    print(f"Starting training for {epochs} epochs...")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=10)

    # 6. Save
    print(f"Saving model to {output_path}")
    model.save(output_path)

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs("./models", exist_ok=True)
    fine_tune_specter()
