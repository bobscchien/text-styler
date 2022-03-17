from utils.tokenizer import *
from utils.servitization import HF2TFSeq2SeqPipeline
from utils.visualization import plot_attention_weights
from models.transformer_bert import HFSelectTokenizer

import os
import streamlit as st

# Setup arguments
text_preprocessors = {'inp':zh_preprocess, 'tar':zh_preprocess}
predictor_dir = '../models/savedmodels/ZH-ZHC_BertEncoderTransformer'
pretrain_dir = '../models/external/'

### Setup model

@st.cache(allow_output_mutation=True)
def model_pipeline(input_text):

    # Build pipeline
    pipeline = HF2TFSeq2SeqPipeline(predictor_dir, pretrain_dir, text_preprocessors)
    bert_name = pipeline.inp_bert
    target_text, target_tokens, attention_weights = pipeline(input_text, return_attention=True)

    # Postprocess
    target_text = target_text.replace(' ', '').replace('[]', '')

    return bert_name, target_text, target_tokens, attention_weights

### Setup visualization

@st.cache()
def attention_plotter(bert_name, user_input, target_tokens, attention_weights, layer, head):

    cache_dir = os.path.join(pretrain_dir, bert_name)
    
    tokenizers = tf.Module()
    tokenizers.inp = HFSelectTokenizer(bert_name).from_pretrained(bert_name, 
                                                                  cache_dir=cache_dir, 
                                                                  do_lower_case=True)

    fig = plot_attention_weights(user_input, target_tokens, attention_weights, tokenizers, layer, head, show=False)
    return fig

### Run application

st.title("Text Style Transfer")
st.header('Traditional Chinese ⇨ Classical Chinese')

st.write('')
user_input = st.text_area('Please input your Traditional Chinese sentence:', 
                          value='父親是一個胖子，走過去自然要費事些。', 
                          max_chars=128)

if st.button('Transfer Text'):
    if bool(user_input):
        bert_name, target_text, target_tokens, attention_weights = model_pipeline(user_input)
        st.write('The result of your Classical Chinese sentence:')
        st.write(target_text)

        #fig = attention_plotter(bert_name, user_input, target_tokens, attention_weights, layer=1, head=1)
        #st.pyplot(fig)
    else:
        st.write('Input sentence cannot be empty. Please say something to me.')
