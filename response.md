1. What should you do if the two models have different tokenizers?

If the two models have different tokenizers, one solution is to generate tokens with one tokenizer, then convert the tokens into the token space of the other model. This can be done by converting the original tokens into natural language, then retokenizing it into the token space of the other model. 

Another possibly more efficient option is to build a mapping between the two token spaces that can translate between the two tokenizers. There are likely tokens that directly match between the two token spaces and the mapping is direct and trivial. For non direct matches, the tokens can be decomposed or transformed in some way into sub-tokens and can further be mapped. 

2. Do you think contrastive decoding is used in practice?

Contrastive decoding seems more computationally intensive since it relies on multiple LLMs, and its use in practice is likely dependent on the scarcity of compute. 