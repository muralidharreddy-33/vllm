# vLLM V1 User Guide

## Why vLLM V1?

vLLM V0 successfully supported a wide range of models and hardware, but as new features were developed independently, the system grew increasingly complex. This complexity made it harder to integrate new capabilities and introduced technical debt, revealing the need for a more streamlined and unified design.

Building on V0’s success, vLLM V1 retains the stable and proven components from V0
(such as the models, GPU kernels, and utilities). At the same time, it significantly
re-architects the core systems, covering the scheduler, KV cache manager, worker,
sampler, and API server, to provide a cohesive, maintainable framework that better
accommodates continued growth and innovation.

Specifically, V1 aims to:

- Provide a **simple, modular, and easy-to-hack codebase**.
- Ensure **high performance** with near-zero CPU overhead.
- **Combine key optimizations** into a unified architecture.
- Require **zero configs** by enabling features/optimizations by default.

For more details, check out the vLLM V1 blog post [vLLM V1: A Major
Upgrade to vLLM’s Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (published Jan 27, 2025).

This living user guide outlines a few known **important changes and limitations** introduced by vLLM V1. The team has been working actively to bring V1 as the default engine version of vLLM, therefore this guide will be updated constantly as more feature get supported on vLLM V1.

### Semantic Changes and Deprecated Features

#### Logprobs

vLLM V1 supports logprobs and prompt logprobs. However, there are some important semantic
differences compared to V0:

**Logprobs Calculation**

Logprobs in V1 are now returned immediately once computed from the model’s raw output (i.e.
before applying any logits post-processing such as temperature scaling or penalty
adjustments). As a result, the returned logprobs do not reflect the final adjusted
probabilities used during sampling.

Support for logprobs with post-sampling adjustments is work in progress and will be added in future updates.

**Prompt Logprobs with Prefix Caching**

Currently prompt logprobs are only supported when prefix caching is turned off via `--no-enable-prefix-caching`. In a future release, prompt logprobs will be compatible with prefix caching, but a recomputation will be triggered to recover the full prompt logprobs even upon a prefix cache hit. See details in [RFC #13414](https://github.com/vllm-project/vllm/issues/13414).

#### Deprecated Features

As part of the major architectural rework in vLLM V1, several legacy features have been deprecated.

**Sampling features**

- **best_of**: This feature has been deprecated due to limited usage. See details at [RFC #13361](https://github.com/vllm-project/vllm/issues/13361).
- **Per-Request Logits Processors**: In V0, users could pass custom
  processing functions to adjust logits on a per-request basis. In vLLM V1 this feature has been deprecated. Instead, the design is moving toward supporting **global logits
  processors**, a feature the team is actively working on for future releases. See details at [RFC #13360](https://github.com/vllm-project/vllm/pull/13360).

**KV Cache features**

- **GPU <> CPU KV Cache Swapping**: with the new simplified core architecture, vLLM V1 no longer requires KV cache swapping
to handle request preemptions.

### Feature & Model Support in Progress

Although the support for many features & models have been re‐implemented or optimized compared to V0, some remain either unsupported or not yet fully optimized on vLLM V1.

#### Unoptimized Features

- **LoRA**: LoRA is functionally working on vLLM V1 but its performance is
  inferior to that of V0. The team is actively working on improving its
  performance
(e.g., see [PR #13096](https://github.com/vllm-project/vllm/pull/13096)).

- **Spec Decode**: Currently, only ngram-based spec decode is supported in V1. There
  will be follow-up work to support other types of spec decode.

#### Unsupported Features

- **FP8 KV Cache**: While vLLM V1 introduces new FP8 kernels for model weight quantization, support for an FP8 key–value cache is not yet available. Users must continue using FP16 (or other supported precisions) for the KV cache.

- **Structured Generation Fallback**: For structured output tasks, V1 currently
  supports only the `xgrammar:no_fallback` mode.
  Details about the structured generation can be found
  [here](https://docs.vllm.ai/en/latest/features/structured_outputs.html).

#### Unsupported Models

vLLM V1 excludes model architectures with the `SupportsV0Only` protocol, and the majority fall into the following categories. V1 support for these models will be added eventually.

**Embedding Models**  
vLLM V1 does not yet include a `PoolingModelRunner` to support embedding/pooling
  models (e.g, `XLMRobertaModel`).

**Mamba Models**  
Models using selective state-space mechanisms (instead of standard transformer attention)
are not yet supported (e.g., `MambaForCausalLM`, `JambaForCausalLM`).

**Encoder-Decoder Models**  
vLLM V1 is currently optimized for decoder-only transformers. Models requiring
  cross-attention between separate encoder and decoder are not yet supported (e.g., `BartForConditionalGeneration`, `MllamaForConditionalGeneration`).

For a complete list of supported models, see the [list of supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## FAQ

TODO
