import wikipedia
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

st.title("BART Text Summarization")
wiki = st.text_input("Search for")

if st.button("Search"):
	wikipage = wikipedia.page(wiki)
	inputs = bart_tokenizer([wikipage.content], max_length=1024, return_tensors='pt')
	summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=100, early_stopping=True)
	summary = ([bart_tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_space=False) for i in summary_ids])
	st.success(summary[0])

