import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"

# Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)
print(indexed_tokens)

# The tokenizer adds special_tokens to represent the start ([CLS]) of a sequence and separation ('[SEP]`) between sentences.
# The tokenizer can break a word down into multiple parts.
# From a linguistic perspective, the second one is interesting. Many languages have word roots, or components that make up a word. For instance, the word "quadratic" has the root "quadr" which means "4". Rather than use word roots as defined by a language, BERT uses a WordPiece model to find patterns in how to break up a word. The BERT model we will be using today has 28996 token vocabulary.

print(tokenizer.convert_ids_to_tokens([str(token) for token in indexed_tokens]))

print(tokenizer.decode(indexed_tokens))

# define which index correspnds to which special token
cls_token = 101
sep_token = 102

def get_segment_ids(indexed_tokens):
    segment_ids = []
    segment_id = 0
    for token in indexed_tokens:
        if token == sep_token:
            segment_id += 1
        segment_ids.append(segment_id)
    segment_ids[-1] -= 1  # Last [SEP] is ignored
    return torch.tensor([segment_ids]), torch.tensor([indexed_tokens])

segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)
print(segments_tensors)

#  BERT masks out a word in a sequence of words. The mask is its own special token
print(tokenizer.mask_token)
print(tokenizer.mask_token_id)

masked_index = 5

indexed_tokens[masked_index] = tokenizer.mask_token_id
tokens_tensor = torch.tensor([indexed_tokens])
print(tokenizer.decode(indexed_tokens))

masked_lm_model = BertForMaskedLM.from_pretrained("bert-base-cased")
print(masked_lm_model)

embedding_table = next(masked_lm_model.bert.embeddings.word_embeddings.parameters())
print(embedding_table)
# embedding of size 768 for each of the 28996 tokens in BERT's vocabulary
print(embedding_table.shape)

# predict the missing word in our provided sentences? We will use torch.no_grad to inform PyTorch not to calculate a gradient
with torch.no_grad():
    predictions = masked_lm_model(tokens_tensor, token_type_ids=segments_tensors)
print(predictions)
print(predictions[0].shape)

# The 24 is our number of tokens, and the 28996 are the predictions for every token in BERT's vocabulary. We'd like to find the highest value accross all the token in the vocabulary, so we can use torch.argmax to find it.

predicted_index = torch.argmax(predictions[0][0], dim=1)[masked_index].item()
print(predicted_index)

predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
print(tokenizer.decode(indexed_tokens))

# While word masking is interesting, BERT was designed for more complex problems such as sentence prediction. It is able to accomplish this by building on the Attention Transformer architecture.
# We will be using a different version of BERT for this section, which has its own tokenizer. Let's find a new set of tokens for our sample sentences.

text_1 = "I understand equations, both the simple and quadratical."
text_2 = "What kind of equations do I understand?"

question_answering_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
segments_tensors, tokens_tensor = get_segment_ids(indexed_tokens)

question_answering_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Predict the start and end positions logits
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)
print(out)

# The question_answering_model and answering model is scanning through our input sequence to find the subsequence that best answers the question. The higher the value, the more likely the start of the answer is.
print(out.start_logits)
print(out.end_logits)

answer_sequence = indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1]
print(answer_sequence)
print(question_answering_tokenizer.convert_ids_to_tokens(answer_sequence))
print(question_answering_tokenizer.decode(answer_sequence))