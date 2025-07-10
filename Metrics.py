import warnings
reference_answer="Startify 3.0 appears to be a goal-setting and habit-tracking platform that helps users achieve their objectives and develop consistent routines."
generated_answer="goal-setting and habit-tracking platform that helps users achieve their objectives and develop consistent routines."

# BLEU score (Bilingual Evalution Understudy) 
import sacrebleu
bleu=sacrebleu.sentence_bleu(generated_answer,[reference_answer])
print(f"BLEU Score:{bleu.score:.4f}")
warnings.filterwarnings("ignore")

# METEOR 
import evaluate
meteor=evaluate.load("meteor")
results=meteor.compute(predictions=[generated_answer],references=[reference_answer])
print(f"METEOR Score:{results['meteor']:.4f}")
warnings.filterwarnings("ignore")

# ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 
from rouge_score import rouge_scorer
scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
scores=scorer.score(reference_answer,generated_answer)
print("ROUGE-1 (unigrams):",scores['rouge1'])
print("ROUGE-2 (bigrams):",scores['rouge2'])
print("ROUGE-L (longest common subsequence):",scores['rougeL'])
warnings.filterwarnings("ignore")

# BERTScore
from bert_score import score
P,R,F1=score([generated_answer],[reference_answer],lang="en",verbose=True)
print(f"BERTScore Precision:{P.item():.4f}")
print(f"BERTScore Recall:{R.item():.4f}")
print(f"BERTScore F1 Score:{F1.item():.4f}")
warnings.filterwarnings("ignore")

# Embedding Similarity
from sentence_transformers import SentenceTransformer, util
model=SentenceTransformer('all-MiniLM-L6-v2')
ref_emb=model.encode(reference_answer,convert_to_tensor=True)
gen_emb=model.encode(generated_answer,convert_to_tensor=True)
similarity=util.cos_sim(ref_emb,gen_emb)
print(f"Embedding Similarity Score:{similarity.item():.4f}")
warnings.filterwarnings("ignore")




'''
BLEU Score:67.0320
METEOR Score:0.7352
ROUGE-1 (unigrams): Score(precision=1.0, recall=0.6956521739130435, fmeasure=0.8205128205128205)
ROUGE-2 (bigrams): Score(precision=1.0, recall=0.6818181818181818, fmeasure=0.8108108108108109)
ROUGE-L (longest common subsequence): Score(precision=1.0, recall=0.6956521739130435, fmeasure=0.8205128205128205)
BERTScore Precision:0.9554
BERTScore Recall:0.9321
BERTScore F1 Score:0.9436
Embedding Similarity Score:0.6642
'''