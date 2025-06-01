Installation

`pip install -r requirements.txt`

setup .env
`GOOGLE_API_KEY="YOU-API` //get on https://aistudio.google.com/app/apikey

Run Indexing Data First (If First or Want Update Data)
`python3 indexing.py`

Update your config.py

`CHUNK_SIZE = 4000 //chunk for data vektor
CHUNK_OVERLAP = 50
RETRIEVAL_K = 16  // document read for AI data`

Run your with streamlit
`streamlit run app.py`
