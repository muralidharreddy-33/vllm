# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List

from vllm.benchmarks.benchmark_throughput import (add_options, main,
                                                  validate_parsed_args)
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class BenchmarkThroughputSubcommand(CLISubcommand):
    """ The `throughput` subcommand for the vllm bench. """

    def __init__(self):
        self.name = "throughput"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_args(args)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser("throughput",
                                       help="Benchmark the throughput.",
                                       usage="vllm bench throughput [options]")
        add_options(parser)
        return parser


def cmd_init() -> List[CLISubcommand]:
    return [BenchmarkThroughputSubcommand()]
