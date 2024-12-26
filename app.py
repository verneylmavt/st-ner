import os
import json
import streamlit as st
import numpy as np
import spacy
import onnx
import onnxruntime

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
Embedding Model = GloVe("6B.200d")
        """,
        "parameters": """
Batch Size = 64
Word Embedding Size = 200
Character Embedding Size = 50
LSTM Hidden Size = 300
Number of LSTM Layers = 2
Number of Attention Heads = 8
Feedforward Hidden Size = 256
Dropout Rate = 0.4
Learning Rate = 0.002071759505536834
Epochs = 15
Optimizer = AdamW
Weight Decay = 0.01
Loss Function = CrossEntropyLoss
Hyperparameter Tuning: Bayesian Optimization
        """,
        "model_code": """
class Model(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, word_embedding_dim, char_embedding_dim,
                 char_hidden_dim, hidden_dim, tagset_size, embeddings=None, dropout=0.5):
        super(Model, self).__init__()
        # Word Embeddings Layer
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=word2idx['<PAD>'])
        if embeddings is not None:
            self.word_embedding.weight = nn.Parameter(embeddings)
            self.word_embedding.weight.requires_grad = True
        self.word_dropout = nn.Dropout(dropout)
        # Character Embeddings Layer
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=char2idx['<PAD>'])
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, num_layers=1,
                                 bidirectional=True, batch_first=True)
        self.char_dropout = nn.Dropout(dropout)
        # BiLSTM Layer
        self.bilstm = nn.LSTM(word_embedding_dim + 2 * char_hidden_dim, hidden_dim // 2,
                              num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        # Normalization Layer
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.lstm_dropout = nn.Dropout(dropout)
        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        # Fully Connected Layer
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    def forward(self, x, chars):
        word_embeds = self.word_embedding(x)
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

# @st.cache_resource
def load_model(model_name):
    try:
        model_path = os.path.join("models", str(model_name), "model-q.onnx")
        net = onnx.load(model_path)
        onnx.checker.check_model(net)
    except FileNotFoundError:
        st.error(f"Model file not found for {model_name}. Please ensure 'model-state.pth' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model for {model_name}: {e}")
        st.stop()
    ort_session = onnxruntime.InferenceSession(model_path)
    return ort_session

# @st.cache_resource
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
        st.error(f"Vocabulary file not found for {model_name}. Please ensure 'vocab.pkl' exists in the model directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the vocabulary for {model_name}: {e}")
        st.stop()

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

def render_ner(tokens, tags):
    tag_colors = {
        "B-PER": "blue",
        "I-PER": "lightblue",
        "B-ORG": "green",
        "I-ORG": "lightgreen",
        "B-LOC": "orange",
        "I-LOC": "lightorange",
        "B-MISC": "purple",
        "I-MISC": "lightpurple",
    }
    
    tag_descriptions = {
        "B-PER": "Beginning of Person",
        "I-PER": "Inside of Person",
        "B-ORG": "Beginning of Organization",
        "I-ORG": " Inside of Organization",
        "B-LOC": "Beginning of Location",
        "I-LOC": "Inside of Location",
        "B-MISC": "Beginning of Miscellaneous",
        "I-MISC": "Inside of Miscellaneous",
    }

    result_str = ""
    for token, tag in zip(tokens, tags):
        if tag in tag_colors:
            color = tag_colors[tag]
            result_str += f":{color}[{token}] "
        else:
            result_str += f"{token} "
    st.write(result_str)

    colored_tags = [x for x in dict.fromkeys(tags) if x != 'O']
    for tag in colored_tags:
         st.write(f":{tag_colors[tag]}[{tag}: {tag_descriptions[tag]}]")


# ----------------------
# Page UI
# ----------------------
def main():
    st.title("Named-Entity Recognition (NER)")
    
    model_names = list(model_info.keys())
    model = st.selectbox("Select a Model", model_names)
    
    word2idx, char2idx, idx2tag  = load_vocab(model)
    net = load_model(model)
    
    st.subheader(model_info[model]["subheader"])
    user_input = st.text_area("Enter Text Here:")
    
    if st.button("Analyze"):
        if user_input.strip():
            with st.spinner('Analyzing...'):
                tokens, tags = predict_ner(net, user_input, word2idx, char2idx, idx2tag)
                render_ner(tokens, tags)
        else:
            st.warning("Please enter some text for analysis.")
            
    st.feedback("thumbs")
    st.warning("""Disclaimer: This model has been quantized for optimization.
            Check here for more details: [GitHub Repo🐙](https://github.com/verneylmavt/st-ner)""")
    st.divider()
    
    st.subheader("""Pre-Processing""")
    st.code(model_info[model]["pre_processing"], language="None")
    
    st.subheader("""Parameters""")
    st.code(model_info[model]["parameters"], language="None")
    
    st.subheader("""Model""")
    st.code(model_info[model]["model_code"], language="python")
    
    if "forward_pass" in model_info[model]:
        st.subheader("Forward Pass")
        for key, value in model_info[model]["forward_pass"].items():
            st.caption(key)
            st.latex(value)
    else: pass

if __name__ == "__main__":
    main()