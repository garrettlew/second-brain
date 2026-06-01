## Paper Summary:

### a) Problem/Motivation 

A substantial proportion of processor performance relies on memory hierarchy performance. Specifically, cache performance has a large effect on the performance of processors. Currently, direct-mapped caches are much faster than set-associative caches when looking at access times for hits, but have more conflict hits due to their lack of associativity.
### b) Technical Approach 

This paper proposes victim caching which is a small fully-associative cache that saves the evicted or victim line from the direct-mapped L1 cache. If the L1 cache misses but hits in the victim cache, the hit line in the victim cache is swapped with the evicted line from the L1 cache. This paper also proposes stream buffering (and multi-way stream buffering), which is storing prefetched data in a separate FIFO buffer instead of the cache. This design means there is no cache pollution, and the prefetching bandwidth is much higher than tagged prefetching, which is one at a time, whereas the stream buffer requests the entire sequential data all at once. 
### c) Contribution over related work

Victim caching improves upon the older "miss caching" technique by changing what gets stored in the secondary buffer. In miss caching, when a block is fetched, it is written to both the L1 and the Miss Cache. Until a conflict actually forces this block out of the L1, the data is sitting in both places at the same time — there is wasted space due to redundancy. Victim caching improves on this by only writing data to the victim cache after it gets kicked out of the L1, guaranteeing that the main cache and the victim cache never hold duplicate data.  
Stream buffering also improved upon previous prefetch techniques like tagged prefetching by introducing a separate buffer to store the prefetched lines instead of the cache, so there isn’t any cache pollution and the prefetch happens at the maximum bandwidth of the L2 cache.
## Design:

1. On a dcache miss, check the victim cache (vcache). If vcache hit, then swap dcache evicted line with vcache line. Else memory request as normal.  
2. On a dcache fill, evict the victim line into vcache.
## Implementation:

1. Add a Cache victim\_cache field to Dcache\_Stage\_struct and init the victim cache in the dcache init  
2. In update\_dcache\_stage(), before calling dcache\_cacheline\_miss(), check the vcache and swap on a hit, otherwise continue as normal  
3. In dcache\_fill\_get\_cacheline(), evict the dcache victim into the vcache.

   Code: [https://github.com/litz-lab/scarab/compare/main...garrettlew:scarab:lab3](https://github.com/litz-lab/scarab/compare/main...garrettlew:scarab:lab3)
## Results:

 ![[lab3_ipc.png]]

![[lab3_dcache_miss.png]]

## Analysis:

There is a slight but pretty negligible decrease in IPC between baseline and the victim cache configuration. The D-cache ratio also has a slight but more noticeable decrease between baseline and the victim cache configuration. Since the vcache is small (only 5 lines), it only helps if the “local” data set fits in the extra 5 lines, so the contribution is small but noticeable. A tiny victim cache essentially provides a proportion of the performance gains of a higher associativity for a fraction of the hardware cost.