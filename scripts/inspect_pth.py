import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--output", default="resume_fixed.ckpt")
args = parser.parse_args()

# Load checkpoint
ckpt = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=False,
)

print(f"Checkpoint epoch: {ckpt['epoch']}")
print(f"Global step: {ckpt['global_step']}")

# Reset EarlyStopping counters
for name, state in ckpt.get("callbacks", {}).items():
    if "EarlyStopping" in name:
        print(f"\nEarlyStopping: {name}")
        print(f"Before: {state}")

        # state["wait_count"] = 0
        # state["stopped_epoch"] = 0

        print(f"Resetting wait_count for {name}")
            
        # state["stopped_epoch"] = 0



        print(f"After: {state}")

# Inspect scheduler state without modifying it
for i, state in enumerate(ckpt.get("lr_schedulers", [])):
    print(f"\nScheduler {i}: {state}")

# Inspect current learning rate
for opt in ckpt.get("optimizer_states", []):
    for group in opt["param_groups"]:
        print(f"Current LR: {group['lr']}")

# Save modified checkpoint
torch.save(ckpt, args.output)

print(f"\nSaved: {args.output}")