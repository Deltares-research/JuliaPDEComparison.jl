
# Evaluation Criteria

This file lists the criteria that we will use to evaluate the Julia packages for PDE solvers. The criteria are designed to assess the suitability of the packages for our specific use case, which involves GPU computing and potential integration with AI models.


## Ease of Use

- **Installation**: How easy is it to install the package and its dependencies? Are there any issues with compatibility or versioning?
- **Initial use**: How easy is it to get started with the package? Are there clear examples and documentation for basic usage?
- **Code clarity**: Can the code be kept clear and understandable while using the package, also when more features are added?

## Performance on CPU and GPU

- **GPU Support**: Does it work well on GPUs? Does it support multiple GPU backends (e.g., CUDA, AMD)?
- **Performance**: How does it perform on CPU and GPU? Are there any benchmarks available?
- **Level of detail**: Does it provide low-level access to GPU features for optimization, or is it more high-level and abstracted?
- **CPU performance**: How does it perform on CPU? Is it competitive with other packages or implementations for CPU-based computations? 
  - **Scalar**: How does it perform on a single CPU core? Is it efficient for small-scale problems or prototyping?
  - **Multi-core**: Does it support multi-threading or distributed computing for larger problems? How does it scale with the number of CPU cores?   

## Support for Automatic Differentiation

- **AD Compatibility**: Does the package support automatic differentiation (AD)? Can it be easily integrated with AD tools for gradient-based optimization and sensitivity analysis?

## Community and Documentation

- **Documentation**: Is the documentation comprehensive and easy to understand? Are there examples and tutorials available?
- **Community**: Is there an active community of users and developers? Are there forums, mailing lists, or other resources for getting help and sharing knowledge?
- **Maintenance**: Is the package actively maintained and updated? Are issues and pull requests addressed in a timely manner?
- **Alignment with our use case**: Does the package align well with our specific use case and requirements, such as support for GPU computing and potential integration with AI models? How likely is it that the package will continue to meet our needs in the future, based on its current development trajectory and community engagement?