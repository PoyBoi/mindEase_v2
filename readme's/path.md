# Path followed of all the things I do

1. Searched for datasets
    - Got them in the folder, labeled
    - Now have to combine them for the use cases:
        1. Mental health prediction
        2. Conversational therapist data
        3. Health monitoring and stats 
2. Find a method to train with the data for prediction and monitoring

# What I have to do (/Have done):
- [ ] Fine-tuning Intent Classification model:
    - Tried going down the RASA path
        - Lack of updated documentation, not enough code to work with for training (CLI Based Training)
    - Tried using SpaCy
        - Import and build issues along with lack of proper code (CLI Training again)
    - Transformers Training
        - Resources:
            - [Using Bi-Directional GAN's with Intent Classification](https://medium.com/@nusfintech.ml/intent-classification-of-texts-using-generative-adversarial-networks-gan-and-bidirectional-29ea2f5ef6a4)
            - [Intent Classification w/Bert](https://hannibunny.github.io/mlbook/transformer/intent_classification_with_bert.html)
        - Model to use:
            - I will probablly go with distilBert because:
                - ```
                    - Speed and Efficiency: DistilBERT is significantly smaller and faster than XLNet, making it more suitable for real-time applications and resource-constrained environments.

                    - Solid Performance: While not always matching XLNet's peak performance, DistilBERT still achieves very strong results on intent classification tasks, often with minimal accuracy loss compared to BERT or XLNet.

                    - Ease of Use: DistilBERT's smaller size makes it easier to fine-tune and deploy, requiring less computational power and training data.
                    ```
            - Going with this instead of XLNet because XLN has a larger size, more training time, and more time that will be taken to perform downstream inference
            - Had 2 options:
                - DistilBert Uncased English, going for this as it's more "neutral"
                - " cased Multi-lingual, not going for this becuase it's supposedly more racist
            - The reason "type id's" are not applicable is because that was one of the improvements of the distil model from the bert model
            - Going with AM(Classification) instead of base AutoModel because AMC provides a classification head and (raw) logits from the model
            - Finished training the model
            - Flatten the 3 column into a 2 column dataset
                - [x] Instead of this, clean up the dataset and make it so that there's a pipeline on what to do
                    - Balanced the dataset as well
                    - [] Need to make a pipeline for the IR to work on
                - Combine the datasets and make the labels universal
                    - See if this is viable post model training (only do this if the output isn't good)

- [ ] Emotion Recognition
    - Need to find the best method to do this
    - Do I use a library or do I fine-tune a model ?

- [ ] Training (Q/LoRA) LLM:
    - Need to use "genuine" conversational data
        - Try using one which has actual therapist data
            - Approach a therapist and ask them the scope and discuss it with them
    - Clean(ed) the data
    - Try to see if fine-tuning a model will be worth the time
        - Try to see if training a model over (MLX) Ollama makes sense

- [ ] Training the Depression Prediction Model (DPM):
    - See how I can extract data from the messages that user sends
        - Need to see what is the legality regarding using user methods
    - Need to make a graph-able dataset that can be shown to the user
    - Need to be able to track their moods (use intent recognition)
        - Or I can just use that dataset which I have for this reason

- [ ] Integrate memory management
    - It should be like the model should develop links between people and their relationship to the user
    - Can be used when the user utters their name later on
        - eg: Lauren -> Mother -> Absusive, Darren -> Father -> Comfort Space
    - How to:
        - Intent Recognition: Your existing intent recognition model continues to classify user intents.
        - Entity Recognition: Implement a Named Entity Recognition (NER) model to identify entities (people, locations, etc.) in the user's text.
        - Memory Linking:
            - When the user introduces a new relationship:
            - Extract the entities and the relationship from the user's input using NER and potentially rule-based methods.
            - Store this relationship in your knowledge graph.
        - When the user mentions an entity:
            - Use NER to identify the entity.
            - Query your knowledge graph to retrieve relevant linked information.
        - Dialogue Management: Your chatbot uses the identified intent and the retrieved memory context to generate a suitable response.