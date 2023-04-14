import argparse
import os


def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, required=True)
    args = parser.parse_args()

    rates = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    duration = args.duration
    for rate in rates:
        cmd = f"python benchmark/benchmark_chatbot.py --dataset sharegpt_clean_lang_10k_opt_tokenized.pkl --model facebook/opt-13b --request-rate {rate} --duration {duration} --n1 1.0 --use-dummy"
        run_cmd(cmd)
