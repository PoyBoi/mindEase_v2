1. Building the chatbot
    - [x] Need to Get a dataset going
      - [x] Need to merge all the other datasets we found into one big dataset with the same formatting (cannot be unstructured)
    - Train Lora / fine tune the model (LLAMA 3.1 8B / Nous Hermes 3)
        - Learn how to train a lora
        - See if fine tuning is better or if lora training is better
    - To train the model on intent (RASA)
        - Need to introduce an intent classifier
    - See if you can add layer of usage of pyautogen
    - Add a layer of guardrails to prevent the user from missusing the chatbot

2. Deploying the chatbot (the fine tuned model or/and with the lora)
    - Host it via NGROK for trial or put it up on modal 

3. Real time stats logging
    - Make a .txt file in the server so that you can check for the person's stats on mental health
    - Add it via pandas as a view-able DF
    - Can also use tensograph for the same

4. GUI
    - Use GRADIO or use basic HTML CSS with JS

5. More data retreival
    - Add a sentimental analysis running on it
    - Make it so that it's easier to track/guesstimate mental illness
    - Make an out so that once the issue has been diagnosed, give them the data and ask them to consult a therapist
      - Add a contact area to get support (Need to partner with a company for this)

7. MISC
    - Add multi-user support
    - Implement Agentic Method with the lora model
        - [ ] PyAutogen
        - [ ] LangGraph
    - 

# Dataset and Planning:
1. For fine-tuning the model, need to cherry-pick the reponses, both user and AI, on the basis of (each with a rating/score):
    - Helpful
    - Give Guidance
    - Supportive/Uplifting
    - Vagueness/Preciseness of the response (Not telling the person what to do, like you get it)

2. The data which we choose to train with, is a continuous process, learns as it goes.
    - It will learn when it encounters something new, something which is wasn't trained on
    - Will select message based on above criteria

3. Can incorporate agents
    - Need to measure the effectiveness

4. Training for the model will be a batch-process
    - Should have a CI/CD pipeline
    - If the responses are good, send it for in-person testing
    - If not, send the data to "bad data"

# Intent Classification:
- Go with either SpaCy or transformers, see which gives better results
- Clean the dataset
    - Figure out what the connections to the LLM are
    - Can I stack model training on another ? Or do I merge the dataset ?
- Need to figure out to feed data into the mental health prediction and monitoring
- Need to figure out if using `DistilBertTokenizer, DistilBertForSequenceClassification` over `AutoModelForSequenceClassification, AutoTokenizer` is better, or the other way around.

# Training model:
- Can either do LoRA fine tuning
- Or, can use Unsloth to "fine-tune" my model if the performance isn't as good as expected

# To do for now:
1. Give a basic functional model to anshu

# Future Questions:
1. Go with a big AI firm based model, such as GPT, and fine tune it and use it until we get customers.
2. We will need to run async jobs if we want the time taken for response to be minimal, what is the optimal TTR ?
3. We will mostly need to have HttpStreaming enabled for our front end so that the text can actually be streamed and not come in through like a massive blob of text at once.
4. Database / Backend