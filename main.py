import transformers as tr
import multiprocessing as mp
# mp.set_start_method('fork')
import torch
import torch.nn.functional as F
# device = torch.device("mps")

from tqdm import tqdm


amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

amateur_path = "Qwen/Qwen1.5-0.5B"
expert_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# expert_path = "Qwen/Qwen2.5-3B-Instruct"
# amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)


def contrastive_generation(amateur: tr.pipeline, 
                           expert: tr.pipeline,
                           prompt: str,
                           max_tokens: int,
                           adaptive_plausibility: float = .1) -> str:
    def get_probs(model, generated, past):
        with torch.inference_mode():
            if past is None:
                outputs = model.model(generated, use_cache=True)
            else:
                last_token = generated[:, -1].unsqueeze(-1)
                outputs = model.model(last_token, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            return probs

    expert.model.eval()
    amateur.model.eval()
    with torch.inference_mode():
        device = next(amateur.model.parameters()).device
        generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        expert_past = None
        amateur_past = None
        for _ in tqdm(range(max_tokens)):
            expert_probs = get_probs(expert, generated, expert_past)
            amateur_probs = get_probs(amateur, generated, amateur_past)
        
            max_prob = expert_probs.max().item()
            threshold = adaptive_plausibility * torch.exp(torch.tensor(max_prob))

            # create a mask for tokens above the plausibility threshold.
            candidate_mask = torch.exp(expert_probs) >= threshold

            # compute contrastive scores
            contrastive_scores = torch.where(
                candidate_mask,
                expert_probs - amateur_probs,
                torch.tensor(float('-inf')).to(expert_probs.device)
            )

            next_token = contrastive_scores.argmax(dim=-1).unsqueeze(0)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return generated