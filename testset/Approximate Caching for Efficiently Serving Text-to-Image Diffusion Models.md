P1: Problem and Motivation  
What problem does the paper address? Why is this problem important? What is the potential impact of solving it?  

Current diffusion models go through a large number of denoising steps, making them resource intensive and slow. Image generation can be speed up by using more costly GPUs or using way fewer diffusion steps (which sacrifices quality). Decreasing latency without the using the previous two solutions would make high quality image generation more accessible, enabling it at scale.  
  
P2: Key Ideas and System Design  
What are the main ideas proposed in the paper? Summarize your understanding of the system design and key components. Why are these design choices effective?  

The main idea the paper proposes is approximate caching, where the first K denoising steps are skipped, and instead an intermediate noise from a different but similar prompt that's in the cache is used as the starting point. NIRVANA balances between hit-rate, an intermediate state of a prompt existing in the cache, and the K steps that can be skipped -- skip too few and there's no savings; skip too much and the denoising conditioned on the prompt is limited. They also proposed LCBFU is a cache policy that prioritizes storing and keeping the intermediate noise states that are both frequently reused and provide the largest computational savings, maximizing NIRVANA’s efficiency under limited cache space. These designs are effective because they found that intra-session prompts tend to have very similar prompts, and even inter-session prompts are similar, making it likely to have high hit rates in the cache.
  
P3: Strengths and Weaknesses  
What are the strengths and limitations of the work? Consider aspects such as novelty, technical depth, system design, and evaluation quality.  

NIRVANA was shown to significantly reduce GPU computation, image generation latency, and serving costs in real-world text-to-image workloads. Approximate caching is a novel idea taking advantage of their findings that prompts within a session are similar. They also saw that image attributes seem to be set during different denoising steps, justifying why intermediate-state reuse is feasible. 
  
P4: Insights and Possible Improvements  
Is there room for improvement? If so, what ideas do you have for improving the system or extending the work?

Their eviction policy creates holes and NIRVANA does not clean these holes because it would require running the full diffusion process. Possibly, these holes can be filled through some interpolation of the neighboring states in the cache.