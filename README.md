# Private-BERTs: Re-Attention, Ensembling and Unlearning

### Abstract
Recent years have seen extensive research on efficiency in privately trained language models, yet progress remains limited. Differential Privacy (DP) has emerged as a leading algorithmic approach, offering provable guarantees for privacy accounting. While various methods have been explored to meet privacy compliance and defend against membership inference attacks, challenges persist in balancing privacy and utility. This project builds on the work of Ding et al. (2024)\cite{dptransformer}, which introduced a re-attention mechanism and phantom gradient clipping to train differentially private transformers. Although their approach demonstrated promising results on vanilla neural network transformers, we investigate two key questions: (i) Can this solution scale effectively to large language models (LLMs)? and (ii) Does ensembling differentially private models from the same family improve utility?Focusing on BERT and its variants, we also explore certified removal unlearning to compare how different privacy mechanisms affect model utility. Our work aims to provide insights into the scalability and ensemble potential of DP-trained language models while evaluating the trade-offs between privacy guarantees and performance.

Here are the main parts:
1. `robert_lora_privacy.py` and `roberta_lora_w_o_privacy.py`
2. 
