# Object-Tracking-In-C-and-CUDA

This project implements a Template-Matching Object Tracking system using SIMT (Single Instruction, Multiple Threads) architecture with CUDA. The goal was to accelerate a sequential C image processing algorithm by parallelizing key functions. The system processes video frames at multiple resolutions (240p, 480p, 720p) to track a target object by comparing a template image against the video frames using Cosine Similarity.

## Presentation Video Link in Youtube:

## Parallel Algorithms Implemented

To transition from a sequential CPU approach to a parallel GPU approach, the following strategies were implemented:

### 1. Grayscale Conversion 
*   **Sequential Approach:** Iterates through every pixel one by one in a nested loop to calculate luminance.
*   **Parallel Approach:** Utilizes Loop Parallelism (Mapping). A unique CUDA thread is assigned to every single pixel $(x, y)$. All threads execute the RGB-to-Grayscale formula simultaneously. This reduces the time complexity from the sum of all pixel conversions to the time required for a single conversion.

### 2. Similarity Map Generation
*   **Sequential Approach:** Uses a "Sliding Window" algorithm with an Integrated Linear Search. It calculates the cosine similarity for one patch, compares it to the current maximum, and then slides to the next position.
*   **Parallel Approach:** Eliminates the sliding window loop. The kernel launches thousands of threads, where each thread is responsible for calculating the similarity score for a specific patch in the frame. All scores are written to a similarity map index simultaneously.

### 3. Maximum Similarity Reduction
*   **Sequential Approach:** Scans the similarity map linearly to find the highest value.
*   **Parallel Approach:** Uses a Parallel Reduction technique. Threads cooperate within a block to compare values in pairs. This reduces the number of steps required to find the global maximum compared to a linear scan.

### 4. Memory Optimizations
*   **Shared Memory:** Threads cooperate to load the template image into the GPU's on-chip Shared Memory (L1 Cache) once, aiming to reduce repeated reads from slower Global Memory.
*   **Tiled Processing (for 720p):** Because large templates do not fit entirely into shared memory, a "Tiling" strategy was implemented. Threads load small $32\times32$ chunks of the template, process them, synchronize using `__syncthreads()`, and move to the next tile.

---

## Results and Discussion

### Execution Time Comparison
The CUDA implementation demonstrated massive performance gains over the sequential C version across all resolutions.

| Resolution | C Sequential Time (ms) | CUDA Global Mem Time (ms) | CUDA Shared Mem Time (ms) | Speedup (vs C) |
| :--- | :--- | :--- | :--- | :--- |
| **240p** | 37,536.53 | 6.53 | **6.72** | ~5,751x |
| **480p** | 494,972.64 | **77.57** | 79.36 | ~6,381x |
| **720p** | 2,691,901.43 | **401.24** | 414.59 | ~6,708x |

### Discussion of Performance

**1. Parallel Speedup**
The gap between CPU and GPU performance widens as resolution increases. While the C program's execution time grows linearly with the number of pixels, the GPU handles the increased load much more efficiently by processing thousands of pixels in parallel. This resulted in speedups greater than 6000x for high-resolution inputs.

**2. Global vs. Shared Memory Trade-offs**
Contrary to theoretical expectations, the Global Memory version often outperformed the Shared Memory optimizations in this specific implementation (480p and 720p).

*   **240p:** Shared memory was slightly faster because the small template size ($~10$KB) fit easily into the Shared Memory ($64$KB), allowing high occupancy.
*   **480p:** Shared memory was slower. The template size ($~34$KB) was too large, limiting the Streaming Multiprocessor (SM) to only hold 1 active block. This prevented switching to other blocks while waiting for memory, causing the hardware to be idle.
*   **720p (Tiled):** The tiled approach introduced significant overhead. The requirement to use `__syncthreads()` inside nested loops forced threads to stop and wait constantly. Additionally, the complex address arithmetic required for tiling outweighed the benefits of caching.

### Conclusion

This project successfully reduced the execution time of the object-tracking program by implementing parallel processing with CUDA. The GPU implementation significantly outperformed the sequential C version across all image resolutions, achieving speedups of 5751.84x for 240p, 6381.39x for 480p, and 6708.97x for 720p. These improvements were made possible by using multiple threads to simultaneously execute tasks, including grayscale conversion, sliding-window patch evaluation, and parallel reduction for maximum-similarity detection.

However, the application of advanced parallelization techniques, such as shared memory and tiling, produced mixed results. The effectiveness of these strategies relies heavily on specific factors, including template size, available shared memory, synchronization overhead, and GPU occupancy. Careful evaluation is required before incorporating these specific optimizations. Overall, the results demonstrate the usefulness of CUDA and GPU processors for accelerating computationally intensive workloads in the field of computer vision.
