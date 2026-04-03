
# What did we find

- Allocating a new array for the diffusion operator is about 2x slower than reusing a preallocated array and mutating it in-place. This is not surprising, but it's good to have a concrete number for this problem size.
- Adding if statements in the inner loop is much slower than computing the diffusion operator on the interior only and leaving the boundary rows/cols of the output array zero. This is because the if statements prevent vectorization and other optimizations.
- Tullio makes multi-threading easy. This is only faster for larger arrays.
- Sparse arrays are much slower than native or Tullio. They are surprisingly slow compared to dense arrays, even though the diffusion operator is very sparse. This is likely because of the overhead of the sparse array data structure and the fact that we are only doing a few operations per nonzero element?