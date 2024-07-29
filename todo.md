1. Building the chatbot
    - Need to Get a dataset going
      - Need to merge all the other datasets we found into one big dataset with the same formatting (cannot be unstructured)
    - Train Lora/ fine tune the model (LLAMA 3 8B)
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
    - Implement pyautogen with the lora model
    - 
