# /Users/panciut/Downloads/257852_marco_panciera/LM/part_A/summarize_all.py

import os
import json
import glob
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_metrics(metrics, plot_path):
    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    dev_ppl = [m["dev_ppl"] for m in metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, dev_ppl, label='Dev Perplexity')
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def generate_markdown_summary(log_dir, hparams, metrics, log_lines, plot_filename):
    best = min(metrics, key=lambda m: m['dev_ppl'])
    final = metrics[-1]

    md_path = os.path.join(log_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Experiment Summary: `{hparams['EXPERIMENT_NAME']}`\n\n")
        f.write(f"**Timestamp:** {hparams['TIMESTAMP']}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Experiment ID: `{hparams['EXPERIMENT_ID']}`\n")
        f.write(f"- Model: `{hparams['MODEL_TYPE']}`\n")
        f.write(f"- Embedding Size: `{hparams['EMB_SIZE']}`\n")
        f.write(f"- Hidden Size: `{hparams['HID_SIZE']}`\n")
        f.write(f"- Learning Rate: `{hparams['LR']}`\n")
        f.write(f"- Device: `{hparams['DEVICE']}`\n\n")

        f.write("## Performance\n")
        f.write(f"- Final Train Loss: `{final['train_loss']:.4f}`\n")
        f.write(f"- Final Dev Loss: `{final['dev_loss']:.4f}`\n")
        f.write(f"- Final Dev Perplexity: `{final['dev_ppl']:.2f}`\n")
        f.write(f"- Best Epoch: `{best['epoch']}` — PPL: `{best['dev_ppl']:.2f}`\n\n")

        f.write("## Metrics Plot\n")
        f.write(f"![Metrics Plot]({os.path.basename(plot_filename)})\n\n")

        f.write("## Training Log Summary (Last Lines)\n")
        for line in log_lines[-6:]:
            f.write(f"- {line.strip()}\n")

def parse_features_from_name(name):
    return {
        "lstm": "_lstm" in name,
        "adamw": "_adamw" in name,
        "embdrop": "_embdrop" in name,
        "outdrop": "_outdrop" in name
    }

def summarize_experiment(log_dir):
    hparams_path = os.path.join(log_dir, "hparams.json")
    metrics_path = os.path.join(log_dir, "metrics.json")
    log_file_path = next(iter(glob.glob(os.path.join(log_dir, "log_*.txt"))), None)

    if not (os.path.isfile(hparams_path) and os.path.isfile(metrics_path) and log_file_path):
        print(f"⚠️ Skipping {log_dir} — Missing required files.")
        return None

    hparams = load_json(hparams_path)
    metrics = load_json(metrics_path)

    with open(log_file_path, 'r') as f:
        log_lines = [line for line in f if line.startswith("2025") and "SUMMARY" not in line]

    # Save plot
    plot_filename = os.path.join(log_dir, "metrics_plot.png")
    plot_metrics(metrics, plot_filename)

    # Write per-experiment markdown
    generate_markdown_summary(log_dir, hparams, metrics, log_lines, plot_filename)

    # Extract flags
    name = hparams['EXPERIMENT_NAME'].lower()
    flags = parse_features_from_name(name)

    final = metrics[-1]
    best = min(metrics, key=lambda m: m['dev_ppl'])
    rel_path = os.path.basename(log_dir)

    return {
        "name": hparams['EXPERIMENT_NAME'],
        "folder": rel_path,
        "id": hparams['EXPERIMENT_ID'],
        "lr": hparams['LR'],
        "emb": hparams['EMB_SIZE'],
        "hid": hparams['HID_SIZE'],
        "timestamp": hparams['TIMESTAMP'],
        "log_file": os.path.basename(log_file_path),
        "train_loss": final['train_loss'],
        "dev_loss": final['dev_loss'],
        "dev_ppl": final['dev_ppl'],
        "best_epoch": best['epoch'],
        "best_ppl": best['dev_ppl'],
        "lstm": flags["lstm"],
        "adamw": flags["adamw"],
        "embdrop": flags["embdrop"],
        "outdrop": flags["outdrop"]
    }

def write_global_summary(all_results, output_path):
    with open(output_path, 'w') as f:
        f.write("# 📊 Global Experiment Summary\n\n")
        f.write("Each row summarizes one experiment. Click on the summary link for full detail.\n\n")

        f.write("| ID | Experiment | Timestamp | Log File | LR | Emb | Hid | LSTM | AdamW | EmbDrop | OutDrop | Final Loss | Best Epoch | Best PPL | Plot | Summary |\n")
        f.write("|----|------------|-----------|----------|----|-----|-----|------|-------|---------|---------|------------|------------|----------|------|---------|\n")

        for r in all_results:
            summary_path = f"{r['folder']}/summary.md"
            plot_path = f"{r['folder']}/metrics_plot.png"
            f.write(
                f"| {r['id']} | `{r['name']}` | {r['timestamp']} | {r['log_file']} | {r['lr']} | {r['emb']} | {r['hid']} | "
                f"{r['lstm']} | {r['adamw']} | {r['embdrop']} | {r['outdrop']} | "
                f"{r['train_loss']:.4f} | {r['best_epoch']} | {r['best_ppl']:.2f} | "
                f"<img src=\"{plot_path}\" width=\"300\"> | [summary]({summary_path}) |\n"
            )

def summarize_all_experiments(base_log_dir):
    print(f"\n🔍 Scanning `{base_log_dir}` for experiment summaries...\n")
    subdirs = [
        os.path.join(base_log_dir, d)
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d))
    ]

    all_results = []
    for subdir in sorted(subdirs):
        res = summarize_experiment(subdir)
        if res:
            all_results.append(res)

    # Sort results by timestamp
    all_results.sort(key=lambda r: r['timestamp'])

    global_summary_path = os.path.join(base_log_dir, "summary_all.md")
    write_global_summary(all_results, global_summary_path)
    print(f"\n📁 Global summary created: {global_summary_path}")

# === MAIN ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", help="Path to log directory containing experiment folders.")
    args = parser.parse_args()

    summarize_all_experiments(args.log_dir)
