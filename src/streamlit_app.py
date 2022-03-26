import os
import configparser

import streamlit as st

from utils.preprocessor import preprocessors, strB2Q
from utils.servitization import HF2TFSeq2SeqPipeline

config = configparser.ConfigParser()

try:
    # local running
    app_local = True
    
    config.read('../config/streamlit.cfg')
    lang = config['data']['lang']
except:
    # github running 
    app_local = False
    
    config.read('/app/text-styler/config/streamlit-deploy.cfg')
    lang = config['data']['lang']
    gdrive_assests_vocab_id   = str(st.secrets["gdrive"]["assests_vocab_id"])
    gdrive_variables_data_id  = str(st.secrets["gdrive"]["variables_data_id"])
    gdrive_variables_index_id = str(st.secrets["gdrive"]["variables_index_id"])
    gdrive_saved_model_id     = str(st.secrets["gdrive"]["saved_model_id"])
    
predictor_dir = config['path']['predictor_dir']

max_lengths = {'inp':config['data'].getint('inp_len'), 'tar':config['data'].getint('tar_len')}
text_preprocessors = {'inp':preprocessors[lang], 'tar':preprocessors[lang]}

### setup page

# https://raw.githubusercontent.com/omnidan/node-emoji/master/lib/emoji.json
st.set_page_config(
    page_title="bobscchien/text-styler",
    page_icon="notebook_with_decorative_cover",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help":None, "Report a Bug":None}
)

### setup functions

@st.cache(allow_output_mutation=True)
def build_model_pipeline(config, text_preprocessors):
    # Build pipeline
    if not app_local:
        from clouds.connect_gdrive import gdown_file_from_google_drive

        # create tmp directories
        os.makedirs(predictor_dir, exist_ok=True)
        os.makedirs(os.path.join(predictor_dir, 'assets'), exist_ok=True)
        os.makedirs(os.path.join(predictor_dir, 'variables'), exist_ok=True)
        
        # setup checkpoint to avoid reloading
        file_vocab = os.path.join(predictor_dir, f"assets/{lang}_vocab.txt")
        file_data  = os.path.join(predictor_dir, "variables/variables.data-00000-of-00001")
        file_index = os.path.join(predictor_dir, "variables/variables.index")
        file_model = os.path.join(predictor_dir, "saved_model.pb")
        
        for file, gdrive_id in zip(
            [
                file_vocab, file_data, file_index, file_model
            ],
            [
                gdrive_assests_vocab_id, gdrive_variables_data_id,
                gdrive_variables_index_id, gdrive_saved_model_id    
            ]
        ):
            if os.path.isfile(file): continue
            gdown_file_from_google_drive(gdrive_id, file)
            #download_file_from_google_drive(gdrive_id, file)

    pipeline = HF2TFSeq2SeqPipeline(predictor_dir, config['path']['pretrain_dir'], text_preprocessors)
    bert_name = pipeline.inp_bert
    return pipeline, bert_name

def model_inference(pipeline, input_text):
    # Predict
    target_text, target_tokens, attention_weights = pipeline(input_text, 
                                                             max_lengths=max_lengths, 
                                                             return_attention=True)

    # Postprocess
    target_text = target_text.replace(' ', '').replace('[]', '')

    return target_text, target_tokens, attention_weights

### setup information

st.title("Text Style Transfer")
st.subheader('')

col_left, col_right = st.columns([1, 1])

text = (
    'This project structures an end-to-end Transformer model using [**Hugging Face**](https://huggingface.co/) & [**Tensorflow**](https://www.tensorflow.org/?hl=zh-tw), '
    'which is composed of the pretrained bert tokenizer & encoder and the [customized tokenizer](https://github.com/bobscchien/text-tokenizer) & decoder, '
    'to get a text styler. '
    '[_**Source Code**_](https://github.com/bobscchien/text-styler)'
)
col_left.markdown(text)

st.write('')
st.subheader('**Traditional Chinese ⇨ Classical Chinese**')

# load model
pipeline, bert_name = build_model_pipeline(config, text_preprocessors)

# user input

text_sample = {}
text_sample[1] = (
    "朋友買了一件衣料，綠色的底子帶白色方格，當她拿給我們看時，一位對圍棋十分感與趣的同學說：「啊，好像棋盤似的。」"
    "「我看倒有點像稿紙。」我說。「真像一塊塊綠豆糕。」一位外號叫「大食客」的同學緊接著說。"
    "我們不禁哄堂大笑，同樣的一件衣料，每個人卻有不同的感覺。"
    "那位朋友連忙把衣料用紙包好，她覺得衣料就是衣料，不是棋盤，也不是稿紙，更不是綠豆糕。"
)
text_sample[2] = (
    "把一隻貓關在一個封閉的鐵容器裏面，並且裝置以下儀器（注意必須確保這儀器不被容器中的貓直接干擾）："
    "在一台蓋格計數器內置入極少量放射性物質，在一小時內，這個放射性物質至少有一個原子衰變的機率為50%，它沒有任何原子衰變的機率也同樣為50%；"
    "假若衰變事件發生了，則蓋格計數管會放電，通過繼電器啟動一個榔頭，榔頭會打破裝有氰化氫的燒瓶。經過一小時以後，假若沒有發生衰變事件，則貓仍舊存活；"
    "否則發生衰變，這套機構被觸發，氰化氫揮發，導致貓隨即死亡。"
    "用以描述整個事件的波函數竟然表達出了活貓與死貓各半糾合在一起的狀態。"
)
text_sample[3] = (
    "JoJo的奇妙冒險中出現了大量的超自然元素，也結合了真實世界的人物和事件。"
    "第一部的故事圍繞著一個只要沾上血就能將配戴者變成吸血鬼的石鬼面，而吸血鬼可以將人變成殭屍，"
    "吸血鬼和殭屍只能被太陽光或波紋氣功消滅，波紋是一種透過規律呼吸來產生能量的武術。"
    "在第二部中，出現了超古代生物柱之男，他們有著遠超過人類、吸血鬼和殭屍的力量以及壽命，但一樣不能曬到太陽光和被紫外線照射。"
    "此外、第二部還使用了納粹、人體實驗、賽博格等元素。"
)

col_l, col_r = st.columns([1, 9])
num = col_l.selectbox('Sample Text', 
                      list(text_sample.keys()))
user_input = col_r.text_area(
    "Input your text here:",
    text_sample[num] if num else "",
    height=150,
    max_chars=max_lengths['inp']
)

if st.button('Transfer Text'):
    if bool(user_input):
        target_text = model_inference(pipeline, user_input)[0]
        target_text = strB2Q(target_text)
        st.write('The result of your Classical Chinese sentence:')
        st.write(target_text)
    else:
        st.write('Input sentence cannot be empty. Please say something to me.')
