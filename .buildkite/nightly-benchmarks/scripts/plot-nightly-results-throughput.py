import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        'Parse command line arguments for summary-nightly-results script.')
    parser.add_argument('--results-folder',
                        type=str,
                        required=True,
                        help='The folder where the results are stored.')
    parser.add_argument('--description',
                        type=str,
                        required=True,
                        help='Description of the results.')
    args = parser.parse_args()
    return args

    
def get_perf(df, method, model, metric):
    
    means = []
    
    for qps in [2,4,8,16,32,"inf"]:
        target = df['Test name'].str.contains(model)
        target = target & df['Engine'].str.contains(method)
        target = target & df['Test name'].str.contains("qps_" + str(qps))
        filtered_df = df[target]

        if filtered_df.empty:
            means.append(0.)
        else:
            means.append(filtered_df[metric].values[0])

    return np.array(means)

def get_perf_w_std(df, method, model, metric):
    
    if metric in ["TTFT", "Latency", "TPOT", "ITL"]:
        mean = get_perf(df, method, model, "Mean " + metric + " (ms)")
        mean = mean.tolist()
        std = get_perf(df, method, model, "Std " + metric + " (ms)")
        if std.mean() == 0:
            std = None
        success = get_perf(df, method, model, "Successful req.")
        if std is not None:
            std = std / np.sqrt(success)
            std = std.tolist()

    else:
        # assert metric == "Tput"
        # mean = get_perf(df, method, model, "Input Tput (tok/s)") + get_perf(df, method, model, "Output Tput (tok/s)")
        # mean = get_perf(df, method, model, 'Tput (req/s)')
        mean = get_perf(df, method, model, metric)
        mean = mean.tolist()
        std = None

    return mean, std


def main(args):
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file, "r") as f:
            results = results + json.loads(f.read())
    
    for result in results:
        if 'sonnet_512_16' in result['Test name']:
            result['Test name'] = result['Test name'].replace('sonnet_512_16', 'prefill_heavy')
        if 'sonnet_512_256' in result['Test name']:
            result['Test name'] = result['Test name'].replace('sonnet_512_256', 'decode_heavy')

            
    
    # generate markdown table
    df = pd.DataFrame.from_dict(sorted(results, key = lambda x: x['Test name']))
    df['Input tokens per request'] = df['Total input tokens'] / df['Successful req.']
    df['Output tokens per request'] = df['Total output tokens'] / df['Successful req.']
    df['Throughput (req/s)'] = df['Tput (req/s)']

    df2 = df.copy()
    
    
    drop_keys = []
    for key in df.keys():
        if key not in ['Test name',
                       'Mean TTFT (ms)',
                       'Std TTFT (ms)',
                       'Mean TPOT (ms)',
                       'Std TPOT (ms)',
                       'Throughput (req/s)',
                       'Input tokens per request',
                       'Output tokens per request'
                       ]:
            drop_keys.append(key)
    df = df.drop(columns=drop_keys)
    table = tabulate(df, 
                     headers='keys', 
                     tablefmt='pipe', 
                     showindex=False)

    with open(f"nightly_results.md", "w") as f:
        f.write(table)

    df = df2
    df['Throughput'] = df['Throughput (req/s)']
    

    plt.rcParams.update({'font.size': 19})
    # plt.set_cmap("twilight")
    plt.style.use('tableau-colorblind10')

    # plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.subplots_adjust(hspace=1)
    methods = ['vllm', 'sglang', 'lmdeploy', 'trt']
    formal_name = ['vLLM', 'SGLang', 'lmdeploy',  'TRT-LLM']
    ls = ['-', '--', '-.', ':']
    # methods = ['vllm_053post1', 'vllm_055'][::-1]
    # formal_name = ['vLLM v0.5.3', 'vLLM v0.6.0'][::-1]
    # for model in ["llama70B"]:
    for i, model in enumerate(["llama8B", "llama70B"]):
        for j, dataset in enumerate(["sharegpt", "decode_heavy", "prefill_heavy"]):
        # for dataset in ["sharegpt"]:
            # for j, metric in enumerate(["TTFT", "TPOT", "Throughput"]):
            for metric in ["Throughput"]:
                
                my_df = df[df["Test name"].str.contains(dataset)]
                # my_dataset_name = {
                #     "sharegpt": "ShareGPT",
                #     "sonnet_512_256": "Decode-heavy",
                #     "sonnet_512_16": "Prefill-heavy",
                # }[dataset]
                my_dataset_name = dataset
                
                ax = axes[i,j]
                if metric in ["TTFT", "Latency", "TPOT", "ITL"]:
                    ax.set_ylabel(f"{metric} (ms)")
                else:
                    ax.set_ylabel(f"Thoughput (req/s)")

                ax.set_title({
                    "sharegpt": "ShareGPT",
                    "decode_heavy": "Decode-heavy",
                    "prefill_heavy": "Prefill-heavy",
                }[my_dataset_name] + ", " + {
                    "llama8B": "Llama 8B",
                    "llama70B": "Llama 70B",
                }[model] + ", " + {
                    "llama8B": "1xH100",
                    "llama70B": "4xH100",
                }[model])
                
                # if metric == "Tput":
                #     ax.set_title(f"{my_dataset_name} Thoughput")
                # else:
                #     ax.set_title(f"{my_dataset_name} {metric}")
                ax.grid(axis='y')
                # print(model, metric)
                
                tput = {}
                for k, method in enumerate(methods):
                    mean, std = get_perf_w_std(my_df, method, model, metric)
                    label = formal_name[k]
                    
                    # print(method, metric, mean, std)
                    
                    if "Tput" not in metric and "Throughput" not in metric:
                        # mean = mean[:-1]
                        # std = std[:-1]
                        ax.errorbar(range(len(mean)),
                                    mean, 
                                    yerr=std, 
                                    capsize=10, 
                                    capthick=4,
                                    label=label,
                                    linestyle=ls[k],
                                    lw=6,)
                        
                        ax.set_xticks(range(len(mean)))
                        ax.set_xticklabels(["2", "4", "8", "16", "32", "inf"])
                        ax.set_xlabel("QPS")
                    else:
                        tput[method] = mean[-1]
                        
                if "Tput" not in metric and "Throughput" not in metric:
                    ax.legend(framealpha=0.5) 
                    ax.set_ylim(bottom=0)  
                else:
                    for _ in range(len(formal_name)):
                        ax.bar(_, tput[methods[_]])
                    ax.set_xticks(range(len(formal_name)))
                    ax.set_xticklabels(formal_name)
                    ax.set_ylim(bottom=0)
                    # ax.bar(formal_name, tput.values())

                

            


    fig.tight_layout()
    fig.savefig(f"nightly_results.png", bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
