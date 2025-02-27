# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List

from vllm.benchmarks.benchmark_latency import add_options, main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkLatencySubcommand(CLISubcommand):
    """ The `benchmark-latency` subcommand for the vLLM CLI. """

    def __init__(self):
        self.name = "benchmark-latency"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "benchmark-latency",
            help=
            "Benchmark the latency of processing a single batch of requests "
            "till completion.",
            usage="vllm benchmark-latency [options]")
        add_options(parser)
        return parser


def cmd_init() -> List[CLISubcommand]:
    return [BenchmarkLatencySubcommand()]
