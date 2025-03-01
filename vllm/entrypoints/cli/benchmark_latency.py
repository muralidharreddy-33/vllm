# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List

from vllm.benchmarks.benchmark_latency import add_options, main
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkLatencySubcommand(CLISubcommand):
    """ The `latency` subcommand for the vllm bench. """

    def __init__(self):
        self.name = "latency"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            "latency",
            help=
            "Benchmark the latency of processing a single batch of requests "
            "till completion.",
            usage="vllm bench latency [options]")
        add_options(parser)
        return parser


def cmd_init() -> List[CLISubcommand]:
    return [BenchmarkLatencySubcommand()]
