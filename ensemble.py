import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
df.dropna(inplace=True)
df = df.iloc[:300001,:]

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

# removing contractions
contractions = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
    "how's": "how is", "I’d": "I would", "I'll": "I will", "I'm": "I am", "I've": "I have", "I’m": "I am",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mightn't": "might not", "mustn't": "must not", "needn't": "need not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not", "that'd": "that would", "that'll": "that will",
    "that's": "that is", "there'd": "there would", "there'll": "there will", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not", "weren't": "were not",
    "what'll": "what will", "what's": "what is", "what've": "what have", "where'd": "where did", "where'll": "where will",
    "where's": "where is", "who'd": "who would", "who'll": "who will", "who's": "who is", "who've": "who have",
    "why'd": "why did", "why'll": "why will", "why's": "why is", "won't": "will not", "wouldn't": "would not",
}
def expand_contractions(text):
    # Check if the input is a float
    if isinstance(text, float):
        return text

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    return text

# apply
df['text'] = df['text'].apply(expand_contractions)

# removing puntuations
df = df.replace(r'[^\w\s]', '', regex=True)

# stop words removal
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# for all col
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop]) if x is not None else '')

# stemming
nltk.download('punkt_tab')
stemmer = PorterStemmer()

def stem_text(text):
    if isinstance(text, str):
        words = nltk.word_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)


df['text'] = df['text'].apply(stem_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42,stratify=df['label']
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)

state_dict = torch.load('/content/drive/MyDrive/Umich/ml/bert_lora_privacy.pt', map_location=torch.device('cpu'))

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('_module.'):
        new_state_dict[k[len('_module.'):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
roberta_model = get_peft_model(roberta_model, lora_config)
state_dict = torch.load('/content/drive/MyDrive/Umich/ml/roberta_lora_privacy.pt', map_location=torch.device('cpu'))

new_state_dict = {k[len('_module.'):] if k.startswith('_module.') else k: v for k, v in state_dict.items()}

roberta_model.load_state_dict(new_state_dict, strict=False)
roberta_model.eval()
roberta_model.to(device)

distil_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],  # DistilBERT naming
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

distil_model = get_peft_model(distil_model, lora_config)

state_dict = torch.load('/content/drive/MyDrive/Umich/ml/distilbert_lora_privacy.pt', map_location=torch.device('cpu'))
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('_module.'):
        new_state_dict[k[len('_module.'):]] = v
    else:
        new_state_dict[k] = v

distil_model.load_state_dict(new_state_dict, strict=False)

distil_model.eval()
distil_model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

input_ids = encodings['input_ids'].to(device)
attention_mask = encodings['attention_mask'].to(device)

tokenizer1 = RobertaTokenizer.from_pretrained('roberta-base')

encodings1 = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

input_ids1 = encodings['input_ids'].to(device)
attention_mask1 = encodings['attention_mask'].to(device)


tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings2 = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)

input_ids2 = encodings['input_ids'].to(device)
attention_mask2 = encodings['attention_mask'].to(device)

input_ids = [input_ids, input_ids1, input_ids2]
attention_mask = [attention_mask, attention_mask1, attention_mask2]

def predict_ensemble(input_ids, attention_mask, weight_distil=0.6):
    with torch.no_grad():
        logits_bert = model(input_ids[0].to(device), attention_mask=attention_mask[0].to(device)).logits
        logits_distil = distil_model(input_ids[2].to(device), attention_mask=attention_mask[2].to(device)).logits
        logits_roberta = roberta_model(input_ids[1].to(device), attention_mask=attention_mask[1].to(device)).logits

        probs_bert = F.softmax(logits_bert, dim=1)
        probs_distil = F.softmax(logits_distil, dim=1)
        probs_roberta = F.softmax(logits_roberta, dim=1)

        avg_probs = (probs_bert + probs_distil + probs_roberta) / 3

        final_preds = torch.argmax(avg_probs, dim=1)
        return final_preds

preds_ensemble = predict_ensemble(input_ids, attention_mask)
true_labels = y_test.values
preds_numpy = preds_ensemble.cpu().numpy()

accuracy_ensemble = accuracy_score(true_labels, preds_numpy)
print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")

