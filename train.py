import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import argparse
from tqdm import tqdm

from model import TransformerNetwork
from data import CharDataset
from inference import inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to data file")
    parser.add_argument("--batch_size", type=int, required=True, help="Size of batches")
    parser.add_argument("--context_window", type=int, required=True, help="Length of context window")
    parser.add_argument("--d_model", type=int, required=True, help="Dimensionality of embeddings")
    parser.add_argument("--n_head", type=int, required=True, help="Number of heads in the MultiHeadAttention Block")
    parser.add_argument("--n_layer", type=int, required=True, help="Number of transformer layers in the network")
    parser.add_argument("--learning_rate", type=float, required=False, default=3e-4, help="Learning rate for training")
    parser.add_argument("--n_epochs", type=int, required=True, help="Number of epochs to run training")
    parser.add_argument("--feedback_frequency", type=int, required=True, help="Number of iterations for per update log")
    parser.add_argument("--max_new_generation_tokens", type=int, required=False, default=128, help="Number of new tokens to generate")
    args = parser.parse_args()

    dataset = CharDataset(args.path, args.context_window)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Batch Size:     {args.batch_size}")
    print(f"Context Window: {args.context_window}")
    print(f"d_model:        {args.d_model}")
    print(f"n_head:         {args.n_head}")
    print(f"n_layer:        {args.n_layer}")
    print(f"Learning Rate:  {args.learning_rate}")
    print(f"Epochs:         {args.n_epochs}")
    print(f"Feedback Iters: {args.feedback_frequency}")

    model = TransformerNetwork(args.context_window, args.d_model, args.n_head, args.n_layer, dataset.vocab_size, mask=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    scaler = GradScaler("cuda") if device == "cuda" else None

    for epoch in range(args.n_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.n_epochs}")
        epoch_loss = 0.0
        feedback_loss = 0.0
        best_loss = float("inf")

        for iter, (samples, targets) in enumerate(loop):
            samples, targets = samples.to(device), targets.to(device)
            optimizer.zero_grad()

            if device == "cuda":
                with autocast("cuda"):
                    logits = model(samples)

                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)

                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(samples)

                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            feedback_loss += loss.item()

            if (iter + 1) % args.feedback_frequency == 0:
                avg_feedback_loss = feedback_loss / args.feedback_frequency
                loop.set_postfix(loss=avg_feedback_loss)
                feedback_loss = 0.0

                model.eval()
                gen = inference(model, dataset, "We are accounted", args.max_new_generation_tokens)
                print(f"\n[Sample Generation]\n{gen}\n")
                model.train()

                if avg_feedback_loss < best_loss:
                    best_loss = avg_feedback_loss
                    torch.save(model.state_dict(), args.save_path)
                    print(f"Saved model with best loss {best_loss:.4f} at {args.save_path}")

        print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    main()