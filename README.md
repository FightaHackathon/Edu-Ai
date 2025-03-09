# Edu-Ai
AI that can answer user's queries based on the user's given pdf, url links and straight direct researchers

It has three main functions , PDF-ChatBot, Edu-Scraper, and Edu Researcher.

## Installation 

1. ```cd Edu-Ai```

2. ```pip install requirements.txt```

3. ```pip install streamlit```

4. ```streamlit run Hellopage.py```

5. Please get your own hugging face token to add in secrets.toml so that you can use the models.

## Files

1. ```Hellopage.py``` : The starting page where you can choose the options between PDF-ChatBot, Edu-Scraper, Edu-Researcher.

2. ```pdf_rag.py``` : Using RAG, user can input their queries and pdfs so that they can recieve their desired answers.

3. ```Edu-Scraper.py``` : Using RAG and selenium, user can input their url and their queries so that they can recieve their desired answers.

4. ```Edu-Researcher.py``` : Using DDGS and Stategraph, user can get straight to the point searches.


### Important notes

1. Users can choose between qwen and deepseek for their preference modes(Can add more models by just modifying the codes slightly

2. Please get your own hugging face token from making an account in https://huggingface.co/welcome

