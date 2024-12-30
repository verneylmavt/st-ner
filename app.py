import os
import json
import streamlit as st
from streamlit_extras.chart_container import chart_container
from streamlit_extras.mention import mention
from streamlit_extras.echo_expander import echo_expander
import numpy as np
import pandas as pd
import spacy
import onnxruntime
from annotated_text import annotated_text

# ----------------------
# Configuration
# ----------------------

# @st.cache_resource
def load_spacy():
    model_path = os.path.join("data", "en_core_web_sm", "en_core_web_sm-3.8.0")
    if os.path.isdir(model_path):
        return spacy.load(model_path)
    else:
        raise FileNotFoundError(f"SpaCy model not found at {model_path}. Please ensure it is correctly placed.")
spacy_en = load_spacy()

# ----------------------
# Model Information
# ----------------------
model_info = {
    "bilstm-w-a": {
        "subheader": "Model: Bi-LSTM w/ Attention",
        "pre_processing": """
Dataset = CoNLL 2003
Tokenizer = NLTK("Word Tokenizer")
Embedding Model = GloVe("6B.200d")
        """,
        "parameters": """
Batch Size = 64

Vocabulary Size = 10,000
Character Vocabulary Size = 87
Word Embedding Dimension = 200
Character Embedding Dimension = 50
Character Hidden Dimension = 50
Hidden Dimension = 256
Tagset Size = 9
Dropout Rate = 0.4

Epochs = 15
Learning Rate = 0.002071759505536834
Loss Function = CrossEntropyLoss
Optimizer = AdamW
Weight Decay = 0.01
Hyperparameter Tuning: Bayesian Optimization
        """,
        "model_code": """
class Model(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, word_embedding_dim, char_embedding_dim,
                 char_hidden_dim, hidden_dim, tagset_size, embeddings=None, dropout=0.5):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=word2idx['<PAD>'])
        if embeddings is not None:
            self.word_embedding.weight = nn.Parameter(embeddings)
            self.word_embedding.weight.requires_grad = True
        self.word_dropout = nn.Dropout(dropout)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=char2idx['<PAD>'])
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=1,
                                 bidirectional=True, batch_first=True)
        self.char_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(word_embedding_dim + 2 * char_hidden_dim, hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lstm_dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, words, chars):
        word_embeds = self.word_embedding(words)
        word_embeds = self.word_dropout(word_embeds)
        batch_size, seq_len, char_len = chars.size()
        chars = chars.view(batch_size * seq_len, char_len)
        char_embeds = self.char_embedding(chars)
        char_lstm_out, _ = self.char_lstm(char_embeds)
        char_hidden = torch.cat((char_lstm_out[:, -1, :char_lstm_out.size(2)//2],
                                 char_lstm_out[:, -1, char_lstm_out.size(2)//2:]), dim=1)
        char_hidden = self.char_dropout(char_hidden)
        char_hidden = char_hidden.view(batch_size, seq_len, -1)
        combined = torch.cat((word_embeds, char_hidden), dim=2)
        lstm_out, _ = self.bilstm(combined)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        combined_output = lstm_out + attn_output
        tag_space = self.hidden2tag(combined_output)
        return tag_space

        """
    }
}

# ----------------------
# Loading Function
# ----------------------

@st.cache_resource
def load_model(model_name):
    try:
        model_path = os.path.join("models", str(model_name), "model-q.onnx")
        ort_session = onnxruntime.InferenceSession(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}. Please ensure 'model-state.pth' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    return ort_session

@st.cache_data
def load_vocab(model_name):
    try:
        model_path = os.path.join("models", model_name)
        with open(os.path.join(model_path, 'word2idx.json'), 'r') as json_file:
            word2idx = json.load(json_file)
        with open(os.path.join(model_path, 'char2idx.json'), 'r') as json_file:
            char2idx = json.load(json_file)
        with open(os.path.join(model_path, 'idx2tag.json'), 'r') as json_file:
            idx2tag = json.load(json_file)
        return word2idx, char2idx, {int(k): v for k, v in idx2tag.items()}
    except FileNotFoundError:
        st.error(f"Vocabulary file not found for {model_name}. Please ensure 'word2idx.json', 'char2idx.json', 'idx2tag.json' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the vocabulary for {model_name}: {e}")
        st.stop()
        
@st.cache_data
def load_training_data():
    training_data = {
    "Epoch": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Train Loss": [0.2450, 0.0729, 0.0474, 0.0354, 0.0250, 0.0192, 0.0157, 0.0134, 0.0107],
    "Train Accuracy": [0.9299, 0.9789, 0.9864, 0.9898, 0.9926, 0.9943, 0.9953, 0.9961, 0.9969],
    "Validation Accuracy": [0.9754, 0.9793, 0.9804, 0.9826, 0.9827, 0.9836, 0.9827, 0.9772, 0.9781],
    }
    return pd.DataFrame(training_data)

# ----------------------
# Prediction Function
# ----------------------
def tokenizer(sentence):
    return [
        token.text for token in spacy_en(sentence)
    ]

def predict_ner(ort_session, sentence, word2idx, char2idx, idx2tag):
    tokens = tokenizer(sentence)
    max_len = 50
    max_char_len = 10
    word_ids = []
    char_ids = []

    for token in tokens:
        word_id = word2idx.get(token.lower(), word2idx['<UNK>'])
        word_ids.append(word_id)
        chars_of_token = [char2idx.get(c, char2idx['<UNK>']) for c in token]
        if len(chars_of_token) > max_char_len:
            chars_of_token = chars_of_token[:max_char_len]
        else:
            chars_of_token += [char2idx['<PAD>']] * (max_char_len - len(chars_of_token))

        char_ids.append(chars_of_token)

    if len(word_ids) > max_len:
        word_ids = word_ids[:max_len]
        char_ids = char_ids[:max_len]
    else:
        pad_length = max_len - len(word_ids)
        word_ids += [word2idx['<PAD>']] * pad_length
        char_ids += [[char2idx['<PAD>']] * max_char_len] * pad_length

    word_tensor = np.array([word_ids], dtype=np.int64)
    char_tensor = np.array([char_ids], dtype=np.int64)
    inputs = {
        "word_ids": word_tensor,
        "char_ids": char_tensor
    }
    emissions = ort_session.run(None, inputs)[0]
    preds = np.argmax(emissions, axis=2).squeeze(0)
    real_length = min(len(tokens), max_len)
    pred_tags = [idx2tag[preds[i]] for i in range(real_length)]
    return tokens, pred_tags

# def render_ner(tokens, tags):
#     # st.write(tokens, tags)
#     tag_colors = {
#         "B-PER": "blue",
#         "I-PER": "cyan",
#         "B-ORG": "green",
#         "I-ORG": "mint",
#         "B-LOC": "orange",
#         "I-LOC": "peach",
#         "B-MISC": "purple",
#         "I-MISC": "lavender",
#     }
    
#     tag_descriptions = {
#         "B-PER": "Beginning of Person",
#         "I-PER": "Inside of Person",
#         "B-ORG": "Beginning of Organization",
#         "I-ORG": " Inside of Organization",
#         "B-LOC": "Beginning of Location",
#         "I-LOC": "Inside of Location",
#         "B-MISC": "Beginning of Miscellaneous",
#         "I-MISC": "Inside of Miscellaneous",
#     }

#     result_str = ""
#     for token, tag in zip(tokens, tags):
#         if tag in tag_colors:
#             color = tag_colors[tag]
#             result_str += f":{color}[{token}] "
#         else:
#             result_str += f"{token} "
#     st.write(result_str)

#     colored_tags = [x for x in dict.fromkeys(tags) if x != 'O']
#     st.write("Tags:", colored_tags)
#     for tag in colored_tags:
#          st.write(f":{tag_colors[tag]}[{tag}: {tag_descriptions[tag]}]")

# def render_ner(tokens, tags):
#     tag_colors = {
#         "B-PER": "#1E90FF",  # Sky Blue
#         "I-PER": "#00CED1",  # Dark Turquoise
#         "B-ORG": "#32CD32",  # Lime Green
#         "I-ORG": "#98FB98",  # Pale Green
#         "B-LOC": "#FFA500",  # Orange
#         "I-LOC": "#FFDAB9",  # Peach Puff
#         "B-MISC": "#9370DB",  # Medium Purple
#         "I-MISC": "#E6E6FA",  # Lavender
#     }
    
#     tag_descriptions = {
#         "B-PER": "Beginning of Person",
#         "I-PER": "Inside of Person",
#         "B-ORG": "Beginning of Organization",
#         "I-ORG": "Inside of Organization",
#         "B-LOC": "Beginning of Location",
#         "I-LOC": "Inside of Location",
#         "B-MISC": "Beginning of Miscellaneous",
#         "I-MISC": "Inside of Miscellaneous",
#     }

#     result_str = ""
#     for token, tag in zip(tokens, tags):
#         if tag in tag_colors:
#             color = tag_colors[tag]
#             result_str += f"<span style='color:{color};'>{token}</span> "
#         else:
#             result_str += f"{token} "
    
#     st.markdown(result_str, unsafe_allow_html=True)

#     colored_tags = [x for x in dict.fromkeys(tags) if x != 'O']
#     for tag in colored_tags:
#         color = tag_colors[tag]
#         st.markdown(
#             f"<span style='color:{color};'>{tag}: {tag_descriptions[tag]}</span>",
#             unsafe_allow_html=True
#         )

def render_ner(tokens, tags):
    tag_descriptions = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous"
    }
    annotated_output = []
    current_entity = []
    current_label = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_entity:
                annotated_output.append((" ".join(current_entity), tag_descriptions[current_label]))
                current_entity = []
            current_entity.append(token)
            current_label = tag.split("-")[1]
        elif tag.startswith("I-") and current_label == tag.split("-")[1]:
            current_entity.append(token)
        else:
            if current_entity:
                annotated_output.append((" ".join(current_entity), tag_descriptions[current_label]))
                current_entity = []
            annotated_output.append(" " + token + " ")
    if current_entity:
        annotated_output.append((" ".join(current_entity), tag_descriptions[current_label]))
    return annotated_output


# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Named-Entity Recognition (NER)")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    st.divider()
    
    word2idx, char2idx, idx2tag  = load_vocab(model)
    net = load_model(model)
    training_data = load_training_data()
    
    st.subheader(model_info[model]["subheader"])
    
    # user_input = st.text_input("Enter Text Here:")
    # if st.button("Recognize"):
    #     if user_input.strip():
    #         with st.spinner('Recognizing...'):
    #             tokens, tags = predict_ner(net, user_input, word2idx, char2idx, idx2tag)
    #             render_ner(tokens, tags)
    #     else:
    #         st.warning("Please enter some text for analysis.")
    
    with st.form(key="ner_form"):
        user_input = st.text_input("Enter Text Here:")
        st.caption("_e.g. U.N. official Ekeus heads for Baghdad._")
        submit_button = st.form_submit_button(label="Recognize")
        
        if submit_button:
            if user_input.strip():
                with st.spinner('Recognizing...'):
                    tokens, tags = predict_ner(net, user_input, word2idx, char2idx, idx2tag)
                    annotated_output = render_ner(tokens, tags)
                annotated_text(*annotated_output)
            else:
                st.warning("Please enter some text for analysis.")
    
    # st.divider()        
    st.feedback("thumbs")
    st.warning("""Disclaimer: This model has been quantized for optimization.""")
    mention(
            label="GitHub Repo: verneylmavt/st-ner",
            icon="github",
            url="https://github.com/verneylmavt/st-ner"
        )
    mention(
            label="Other ML Tasks",
            icon="streamlit",
            url="https://verneylogyt.streamlit.app/"
        )
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    with echo_expander(code_location="below", label="Code"):
        import torch
        import torch.nn as nn
        
        
        class Model(nn.Module):
            def __init__(self, vocab_size, char_vocab_size, word_embedding_dim, char_embedding_dim,
                        char_hidden_dim, hidden_dim, tagset_size, embeddings=None, dropout=0.5):
                super(Model, self).__init__()
                # Embedding Layer for Word w/ Padding Index for '<PAD>'
                self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=word2idx['<PAD>'])
                # Parameter Layer for Word Embeddings Initialization w/ Pre-Trained Embeddings (if provided)
                if embeddings is not None:
                    self.word_embedding.weight = nn.Parameter(embeddings)
                    self.word_embedding.weight.requires_grad = True # Gradient Enabling for Fine-Tuning
                # Dropout Layer for Word Regularization
                self.word_dropout = nn.Dropout(dropout)
                
                # Embedding Layer for Character w/ Padding Index for '<PAD>'
                self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=char2idx['<PAD>'])
                # UniLSTM Layer for Character Feature Extraction
                self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=1,
                                        bidirectional=True, batch_first=True)
                # Dropout Layer for Character Regularization
                self.char_dropout = nn.Dropout(dropout)
                
                # BiLSTM Layer for Contextual Word and Character Representation
                self.bilstm = nn.LSTM(word_embedding_dim + 2 * char_hidden_dim, hidden_dim // 2,
                                    num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
                # Normalization Layer for Word and Character Stabilization
                self.layer_norm = nn.LayerNorm(hidden_dim)
                # Dropout Layer for Word and Character Regularization
                self.lstm_dropout = nn.Dropout(dropout)
                # Multi-Head Attention Layer for Contextual Word and Character Feature Extraction
                self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
                # Fully Connected Layer for Word and Character → Tag Space
                self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            
            def forward(self, words, chars):
                # Word Embeddings of Input Words
                word_embeds = self.word_embedding(words)
                # Dropout of Word Embeddings
                word_embeds = self.word_dropout(word_embeds)

                batch_size, seq_len, char_len = chars.size()
                chars = chars.view(batch_size * seq_len, char_len)
                # Character Embeddings of Input Characters
                char_embeds = self.char_embedding(chars)
                # UniLSTM of Character Embeddings 
                char_lstm_out, _ = self.char_lstm(char_embeds)
                # Concatenation of Forward and Backward Character Hidden States of Last Time Step
                char_hidden = torch.cat((char_lstm_out[:, -1, :char_lstm_out.size(2)//2],
                                        char_lstm_out[:, -1, char_lstm_out.size(2)//2:]), dim=1)
                # Dropout of Character Hidden States
                char_hidden = self.char_dropout(char_hidden)
                # Reshaping of Character Hidden States → Word Embeddings
                char_hidden = char_hidden.view(batch_size, seq_len, -1)
                
                # Concatenation of Word Embeddings and Character Hidden States
                combined = torch.cat((word_embeds, char_hidden), dim=2)
                # BiLSTM of Word Embeddings and Character Hidden States
                lstm_out, _ = self.bilstm(combined)
                # Normalization of BiLSTM Output
                lstm_out = self.layer_norm(lstm_out)
                # Dropout of BiLSTM Output
                lstm_out = self.lstm_dropout(lstm_out)
                # Attention of BiLSTM Output
                attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
                # Residual Connection: BiLSTM Output + Attention Output
                combined_output = lstm_out + attn_output
                # Transformation of Residual Connection → Tag Space
                tag_space = self.hidden2tag(combined_output)
                return tag_space
    # st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass
    
    st.subheader("""Training""")
    # st.line_chart(training_data.set_index("Epoch"))
    with chart_container(training_data):
        st.line_chart(training_data.set_index("Epoch"))
    
    st.subheader("""Evaluation Metrics""")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "0.9682", border=True)
    col2.metric("Precision", "0.8105", border=True)
    col3.metric("Recall", "0.8433", border=True)
    col4.metric("F1 Score", "0.8266", border=True)

if __name__ == "__main__":
    main()