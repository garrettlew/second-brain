# Paper Summary: Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models

## P1: Problem and Motivation

The paper addresses the high cost and latency of serving text-to-image diffusion models. These models generate high-quality images, but they require many iterative denoising steps, which makes inference slow and expensive on GPUs. This is important because text-to-image systems are becoming widely used in real production services, so even small reductions in latency and GPU usage can lead to large savings. The main impact of solving this problem is making image generation faster, cheaper, and more practical for interactive user-facing applications.

## P2: Key Ideas and System Design

The main idea of the paper is approximate caching. Instead of always starting image generation from random noise, the system reuses an intermediate noise state from a previous, similar prompt and continues denoising from that point. The authors build a system called NIRVANA, which uses prompt embeddings, a vector database, a cache selector, and a cache maintenance policy to decide when and how much computation can be skipped. This design is effective because it does not simply retrieve an old image, it reconditions a cached intermediate state for the new prompt, so it can still generate a new image while saving computation.

## P3: Strengths and Weaknesses

A major strength of this work is that it targets a real systems problem in diffusion model serving and evaluates the solution on large production-style workloads. The paper also studies prompt similarity, cache hit rates, latency, compute savings, and image quality. Another strong point is that NIRVANA keeps image quality close to the vanilla diffusion model while reducing GPU compute, latency, and cost. One limitation is that the approach depends heavily on having similar prompts in the cache, so its benefit may be lower for highly diverse or constantly changing workloads. Also, choosing the right amount of skipped denoising steps is tricky because skipping too much can reduce image quality.

## P4: Insights and Possible Improvements

I think the most interesting insight is that diffusion models have reusable intermediate states, not just reusable final outputs. This makes caching more flexible than simple image retrieval. A possible improvement would be to make the cache selector more adaptive, maybe using a learned model that predicts the best denoising step to reuse based on prompt similarity, image style, and past quality feedback. Another extension could combine NIRVANA with other optimization methods like quantization, batching, or faster samplers to get even more savings while still preserving image quality.
