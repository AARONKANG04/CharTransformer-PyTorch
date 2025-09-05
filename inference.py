import torch
import torch.nn.functional as F

def inference(model, dataset, prompt="We are accounted", max_new_tokens=64, device="cpu"):
    model.eval()
    prompt = torch.tensor([dataset.stoi[ch] for ch in prompt], dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(prompt[..., -model.max_seq_len:])
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        prompt = torch.cat([prompt, next_token], dim=1)

    out = "".join([dataset.itos[int(i)] for i in prompt[0]])
    return out

