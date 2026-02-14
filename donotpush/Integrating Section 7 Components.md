# **Architectural Implementation Strategy for Unified-QEC: Sinter Integration, Model Zoo, and Open Data Compliance**

## **1\. The Strategic Imperative of Section 7 Compliance in Quantum Error Correction**

The rapid evolution of Quantum Error Correction (QEC) from theoretical abstraction to engineering reality necessitates a fundamental shift in the software architectures supporting simulation and analysis. The unified-qec repository, positioned as a central component of the JAX-native HoloJAX SDK , stands at a critical juncture. The requirements outlined in "Section 7" of the governing standards document are not merely administrative checkboxes; they represent a rigorous technical mandate designed to ensure scalability, reproducibility, and interoperability in an era where simulation workloads are approaching exascale proportions. This report provides a comprehensive, expert-level technical blueprint for implementing the three pillars of Section 7: the integration of high-performance Monte Carlo sampling via Sinter, the establishment of a version-controlled Model Zoo using Orbax and Hugging Face, and the adoption of cloud-native Open Data standards through Zarr and WebDataset technologies.  
The contemporary landscape of QEC research is characterized by the need to estimate logical error rates with extreme precision, often requiring the simulation of billions of decoding shots to detect rare failure events. Traditional, ad-hoc Python scripts are no longer sufficient for this task. They suffer from initialization overheads, lack standard output formats, and often fail to utilize modern high-performance computing (HPC) resources efficiently. Section 7 addresses these deficiencies by mandating adherence to the Sinter API, which standardizes the interface between the error model (the problem) and the decoder (the solver). Furthermore, the rise of machine learning (ML) based decoders—specifically those implemented in JAX/Flax—introduces new challenges regarding model artifact management. The "Model Zoo" requirement seeks to resolve the crisis of reproducibility by enforcing strict versioning and programmatic access to pre-trained weights, leveraging the Orbax checkpointing framework and the global distribution capabilities of the Hugging Face Hub. Finally, the "Open Data" requirement recognizes that the sheer volume of syndrome data generated during these simulations creates an I/O bottleneck that legacy formats like HDF5 can no longer address effectively in cloud-native environments, necessitating a shift to sharded, stream-oriented formats.  
This report dissects these requirements into actionable engineering tasks. It explores the nuances of implementing the sinter.Decoder abstract base class to amortize compilation costs, details the transition from legacy Flax checkpointing to the modular Orbax system, and provides a comparative analysis of data formats to justify the architectural decision to split storage strategies between training (WebDataset) and analysis (Zarr) workloads. By executing this roadmap, the unified-qec repository will not only achieve Section 7 compliance but will also establish itself as a robust, future-proof platform for quantum fault tolerance research.

## **2\. Sinter Integration: The High-Performance Compute Engine**

### **2.1 The Architecture of Standardized Monte Carlo Sampling**

The integration of Sinter into unified-qec is the primary mechanism for achieving the throughput required by Section 7\. Sinter acts as a high-performance executive, managing the parallel execution of decoding tasks across available CPU cores while abstracting away the complexities of statistical aggregation, error bar calculation, and worker process management. For a repository like unified-qec, which likely contains custom JAX-based neural decoders, the integration challenge lies in bridging the gap between Sinter's process-based parallelism and JAX's accelerator-based resource management.  
The fundamental operation of Sinter revolves around the concept of a "Task," which pairs a quantum circuit (defining the noise and logical observables) with a decoder. Sinter efficiently samples these tasks until a convergence criterion—such as a target number of errors or shots—is met. This dynamic termination capability is essential for efficiently allocating compute resources, as it prevents over-sampling of high-error regimes while ensuring sufficient statistics are gathered for low-error regimes where logical failures are rare events. To participate in this ecosystem, unified-qec must expose its decoding logic through a class that strictly adheres to the sinter.Decoder interface. This is not a trivial wrapper; it requires a deep understanding of Python's multiprocessing behavior, serialization constraints, and memory management.  
The mandate of Section 7 to support "custom decoders" implies that unified-qec cannot simply rely on the pre-compiled binary decoders like PyMatching or Fusion Blossom for its internal research. Instead, it must wrap its experimental algorithms—whether they are renormalization group decoders, belief propagation variants, or neural networks—into a structure that Sinter can instantiate and execute. This requires the implementation of a class inheriting from sinter.Decoder, utilizing the abc (Abstract Base Class) module to enforce the implementation of required methods. The robust design of this wrapper is critical; a naive implementation that re-initializes heavy resources (like JAX contexts or large lookup tables) for every batch of shots will introduce catastrophic latency, negating the benefits of Sinter's optimized sampling loop.

### **2.2 Implementing the sinter.Decoder Abstract Base Class**

To satisfy the technical requirements of Section 7, the unified-qec repository must define a class—hereafter referred to as UnifiedQECDecoder—that implements the sinter.Decoder contract. This class serves as the bridge between Sinter's C++ optimized sampling loops and the Python-level logic of the custom decoder.

#### **2.2.1 Serialization and Worker Initialization**

One of the most frequent points of failure in Sinter integrations involves the serialization of the decoder object. Sinter uses Python's multiprocessing library to spawn worker processes. Consequently, the sinter.Decoder instance must be pickle-serializable. This constraint has profound implications for unified-qec. Specifically, the \_\_init\_\_ method of the UnifiedQECDecoder must remain lightweight. It should effectively act as a configuration container, storing only standard Python types like strings (paths to model weights), dictionaries (hyperparameters), and primitives. It must **not** attempt to load JAX models, initialize TensorFlow sessions, or allocate GPU memory within \_\_init\_\_. Doing so would cause the pickling process to fail or, worse, lead to undefined behavior when the CUDA context is forked to worker processes.  
The correct architectural pattern dictates that resource-heavy initialization must be deferred until the decoder is actually called within the worker process. This lazy initialization ensures that the main process remains lightweight and responsive, while the heavy lifting is distributed correctly across the compute resources. The UnifiedQECDecoder essentially becomes a factory that knows *how* to build the actual decoding engine, rather than being the engine itself.

#### **2.2.2 The compile\_decoder\_for\_dem Optimization**

The most critical performance requirement in Section 7 is likely the implementation of the compile\_decoder\_for\_dem method. Sinter provides two pathways for decoding: decode\_via\_files and compile\_decoder\_for\_dem. The default fallback, decode\_via\_files, operates by writing detection events to a temporary file on disk, invoking the decoder, and then reading the predictions back from disk. While this method is robust and compatible with any CLI-based tool, the I/O latency it introduces is unacceptable for high-throughput simulation, particularly when the decoding time itself is on the order of microseconds.  
To achieve the necessary throughput, unified-qec must implement compile\_decoder\_for\_dem. This method allows the decoder to inspect the Detector Error Model (DEM) describing the circuit noise and construct an optimized, in-memory representation of the decoding problem. For a neural decoder, this compilation phase involves determining the input tensor shapes based on the number of detectors, loading the model weights from the Model Zoo (discussed in Section 3), and performing JAX's Just-In-Time (JIT) compilation (using jax.jit or pjit) to optimize the execution graph for the specific problem size.  
The method returns a sinter.CompiledDecoder object. This object persists in the memory of the worker process across multiple batches of shots. This architectural choice amortizes the high cost of JAX initialization and graph compilation over millions of subsequent decoding events. Without this implementation, the system would pay the initialization penalty for every batch, leading to performance degradation of several orders of magnitude.

#### **2.2.3 Bit-Packed Communication with decode\_shots\_bit\_packed**

The CompiledDecoder object returned by the compilation step must implement the decode\_shots\_bit\_packed method. This method represents the inner loop of the simulation. Crucially, Sinter passes detection event data to this method not as boolean arrays, but as bit-packed uint8 NumPy arrays. This design choice reduces the memory bandwidth requirements by a factor of 8, which is significant when streaming terabytes of syndrome data.  
For unified-qec's JAX integration, this presents a specific technical challenge. While JAX performs exceptionally well with float operations, its support for bitwise operations on packed integers has historically been less optimized than NumPy's. However, recent updates to JAX have introduced jax.numpy.unpackbits, which allows for efficient unpacking directly on the accelerator (GPU/TPU). The implementation within unified-qec must take the input uint8 array, transfer it to the device memory, and unpack it into the floating-point or boolean tensor format expected by the neural network. Conversely, the decoder's predictions must be bit-packed before being returned to Sinter.  
This cycle of unpack-infer-pack must be heavily optimized. The use of jax.jit on the entire sequence is mandatory to fuse the unpacking kernels with the inference kernels, minimizing the overhead of kernel launches and memory transfers. The rigorous adherence to this bit-packed interface is non-negotiable for Section 7 compliance, as it ensures the decoder does not become the bottleneck in the simulation pipeline.

### **2.3 Benchmarking Against Industry Standards: PyMatching and Fusion Blossom**

Section 7 requirements imply not just the ability to run custom decoders, but to benchmark them against established state-of-the-art (SOTA) baselines. The unified-qec repository must therefore natively support the execution of PyMatching v2 and Fusion Blossom within the same Sinter framework.

#### **2.3.1 PyMatching v2: The Sparse Blossom Standard**

PyMatching v2 utilizes the "sparse blossom" algorithm, a significant advancement over previous MWPM implementations. It achieves decoding speeds compatible with superconducting qubit cycle times (sub-microsecond per round for distance 17). Integrating PyMatching is straightforward as it is the default backend for Sinter. However, unified-qec needs to ensure that it exposes configuration options—such as the choice between standard matching and correlated matching (which handles hyperedge errors)—via the json\_metadata field in Sinter tasks. This allows for a fair comparison where the baseline decoder is tuned to the specific noise model being simulated.

#### **2.3.2 Fusion Blossom: The Parallel Frontier**

Fusion Blossom represents a paradigm shift towards parallelizable decoding, merging Union-Find concepts with MWPM accuracy. Its relevance to Section 7 lies in its ability to scale to extremely large codes where single-threaded decoders become too slow. The integration strategy for Fusion Blossom involves ensuring the package is installed and invoked correctly via Sinter's custom\_decoders mechanism if it is not natively bundled. The benchmarks provided in the research material indicate that Fusion Blossom can maintain linear time complexity relative to the number of detectors, a property that unified-qec's custom decoders should strive to emulate or exceed.

### **2.4 Handling "Internal" and Legacy Decoders**

Research environments often contain legacy "internal" decoders that do not conform to public APIs. Sinter historically supported internal\_correlated decoders, but these are being deprecated in favor of standardized interfaces. For unified-qec to be compliant with Section 7, any such internal logic must be refactored into the sinter.Decoder class structure. This standardization eliminates "black box" dependencies and ensures that all decoding logic is transparent, portable, and reproducible—key tenets of the Open Data initiative.  
The table below summarizes the required decoder integration features for Section 7 compliance.

| Feature | Sinter Default (File-Based) | Section 7 Compliant (Memory-Based) | Impact on Unified-QEC |
| :---- | :---- | :---- | :---- |
| **API Method** | decode\_via\_files | compile\_decoder\_for\_dem | Critical for JAX compilation amortization. |
| **Data Format** | Temporary Files (.b8) | Bit-packed Arrays (uint8) | 8x reduction in memory bandwidth; requires jax.numpy.unpackbits. |
| **Initialization** | Once per batch | Once per noise model | Enables massive batch processing without overhead. |
| **Concurrency** | Process-based | Process \+ Device Queue | Requires careful GPU memory management (client-server model). |
| **State** | Stateless | Stateful (Compiled) | Allows caching of XLA executables and auxiliary graphs. |

## **3\. The Model Zoo: Architecture for Artifact Management and Distribution**

### **3.1 The Reproducibility Crisis and the Model Zoo Solution**

In the context of ML-augmented QEC, a "decoder" is no longer just an algorithm; it is a specific set of neural network weights trained on a specific noise distribution. The "Model Zoo" requirement of Section 7 addresses the reproducibility crisis by mandating a robust system for managing these artifacts. It requires unified-qec to transition from ad-hoc weight files (e.g., loose .npy or .pt files) to a structured, versioned, and programmatic repository system. This system relies on two key technologies: **Orbax** for serialization within the JAX ecosystem, and **Hugging Face Hub** for global distribution and version control.

### **3.2 Advanced Serialization with Orbax**

The shift to JAX necessitates the adoption of Orbax, Google's recommended checkpointing library, replacing the legacy flax.training.checkpoints API. Orbax offers sophisticated features essential for the scale of models anticipated in HoloJAX SDK applications, particularly regarding atomicity and asynchronous saving.

#### **3.2.1 The CheckpointManager Pattern**

The core of the Model Zoo's persistence layer is the orbax.checkpoint.CheckpointManager. Unlike simple file writing, the CheckpointManager handles the complexity of maintaining a history of checkpoints (e.g., keeping the "best" 5 models based on validation accuracy), deleting obsolete files, and ensuring that writes are atomic—preventing corruption if a training job is preempted.  
The implementation within unified-qec must utilize CheckpointManagerOptions to define retention policies.  
`options = ocp.CheckpointManagerOptions(max_to_keep=5, save_interval_steps=1000)`  
`manager = ocp.CheckpointManager(path, options=options)`

This ensures that long-running training campaigns on HPC clusters do not exhaust storage quotas with redundant checkpoints.

#### **3.2.2 Composite Checkpoints and PyTrees**

QEC models are rarely just a single array of weights. They involve complex nested structures (PyTrees) comprising model parameters, optimizer states (like Adam moments), and auxiliary batch statistics. Furthermore, a complete artifact in the Model Zoo requires metadata: the architecture hyperparameters, the noise model description, and the training configuration.  
Orbax handles this complexity through Composite arguments. The unified-qec implementation must save the model state and the metadata as distinct items within a single checkpoint directory.  
`save_args = ocp.args.Composite(`  
    `model=ocp.args.StandardSave(train_state),`  
    `metadata=ocp.args.JsonSave(config_dict)`  
`)`  
`manager.save(step, args=save_args)`

This separation allows a user to restore *only* the configuration to inspect hyperparameters without loading the massive weight tensors into memory, or to load the weights for inference while discarding the optimizer state—a crucial feature for efficient deployment from the Model Zoo.

### **3.3 Distribution and Versioning via Hugging Face Hub**

While Orbax handles the *format* of the files, the *distribution* requirement of Section 7 is satisfied by integrating with the Hugging Face Hub. This platform effectively acts as a "GitHub for Models," providing git-based version control for large binary files (LFS).

#### **3.3.1 Programmatic Upload and Synchronization**

The training pipelines in unified-qec must be instrumented to automatically push checkpoints to the Hub. This creates a live Model Zoo that updates as research progresses. The huggingface\_hub Python library provides the upload\_folder function, which syncs a local directory to a remote repository.  
Crucially, this integration supports "commit" semantics. Each saved checkpoint corresponds to a commit hash. This allows researchers to pinpoint the exact version of a model used in a paper, satisfying the strictest interpretation of reproducibility. The unified-qec repository should implement a callback that triggers this upload at the end of every epoch or successful validation run.

#### **3.3.2 Caching and Revision Pinning**

On the consumer side (i.e., when running Sinter), the repository needs a mechanism to fetch these models efficiently. The hf\_hub\_download and snapshot\_download functions provide this capability, backed by a robust local caching system.  
When a user requests a decoder, the system first checks the local cache (typically \~/.cache/huggingface). If the file is present and matches the requested revision (SHA hash), it is used immediately without network overhead. This "smart caching" is vital for large-scale Monte Carlo simulations where thousands of Sinter workers might be launching simultaneously on a cluster; they can all read from a shared file system cache rather than ddosing the Hugging Face servers.  
The API design for unified-qec should expose the revision parameter to the end-user. This allows a Sinter task to specify not just "the Surface Code Decoder" but "The Surface Code Decoder as it existed on Feb 14, 2026," ensuring that simulation results are immutable over time.

#### **3.3.3 Metadata and Model Cards**

A Model Zoo is useless without documentation. The Hugging Face Hub uses README.md files with YAML metadata headers (Model Cards) to index models. Section 7 compliance requires that unified-qec automatically generates these cards. The metadata must include:

* **Tags:** quantum-error-correction, jax, sinter-compatible.  
* **Hyperparameters:** Code distance, noise model details (p values), network architecture dimensions.  
* **Metrics:** Achieved threshold, logical error rates at varying physical error rates.

Automating the creation of these cards ensures that every artifact in the Zoo is searchable and self-documenting, preventing the repository from becoming a "digital landfill" of unnamed weight files.

## **4\. Open Data Architecture: Cloud-Native Storage Strategies**

The "Open Data" requirement of Section 7 confronts the reality of modern QEC research: data volume. Simulating high-distance codes to detect logical error rates of 10^{-12} requires generating and processing trillions of syndrome shots. Storing this data for analysis or training requires a storage architecture that transcends local file systems.

### **4.1 The Inadequacy of HDF5 for Cloud Workloads**

Historically, HDF5 has been the standard for scientific data. However, it was designed for POSIX file systems and struggles significantly in cloud-native environments (like S3 or Google Cloud Storage). HDF5's monolithic file structure requires fetching large metadata headers to read small chunks of data, and it lacks efficient support for concurrent writes from thousands of distributed workers. For the "HoloJAX" ecosystem, which implies distributed, accelerator-based computing, continuing to rely on HDF5 would create a severe I/O bottleneck.

### **4.2 The Dual-Format Strategy: WebDataset and Zarr**

To satisfy the diverse needs of *training* (streaming) and *analysis* (random access), this report recommends a bifurcated storage strategy for unified-qec.

#### **4.2.1 WebDataset for Massive Training Pipelines**

Training neural decoders is a throughput-bound problem. The data loader needs to feed the GPU with batches of syndromes as fast as the GPU can process them. Random access is not required; in fact, it is detrimental.  
**WebDataset** addresses this by storing data as a sequence of POSIX tar archives (shards), each containing thousands of samples.

* **Streaming Efficiency:** This format allows data to be streamed directly from object storage to the compute node using standard HTTP/S requests. There is no need to download the entire dataset before starting training.  
* **Scalability:** By breaking petabytes of data into manageable shards (e.g., 1GB tar files), WebDataset avoids the "inode exhaustion" problem associated with storing millions of individual image/syndrome files.  
* **JAX Integration:** WebDataset integrates seamlessly with JAX pipelines. Libraries like grain or standard Python iterators can consume these streams, apply shuffling via an in-memory buffer, and batch the data for the TPU.

For Section 7 compliance, unified-qec must implement a pipeline that takes Sinter output and serializes it into WebDataset shards (e.g., shards/surface\_d11\_p001\_000.tar). This ensures that the published Open Data is "training-ready."

#### **4.2.2 Zarr for Analytical Random Access**

While WebDataset is ideal for training, researchers often need to inspect specific subsets of data—for example, "load all failure events where the syndrome weight was less than 4." **Zarr** is the cloud-optimized answer to HDF5. It stores N-dimensional arrays as a hierarchy of compressed chunks, where each chunk is a separate key in the object store.

* **Parallel I/O:** Multiple workers can write to different chunks of a Zarr array simultaneously without locking, resolving HDF5's concurrency issues.  
* **Selective Reading:** A researcher can request a specific slice of the tensor, and Zarr will fetch only the relevant chunks from the cloud, minimizing data transfer.

For the Open Data releases, unified-qec should publish "Gold Standard" evaluation datasets in Zarr format. This facilitates interactive analysis in Jupyter notebooks without requiring the user to download terabytes of training data.

### **4.3 Data Format Comparison Matrix**

The table below highlights why the shift from HDF5 is necessary for Section 7 compliance.

| Feature | HDF5 (Legacy) | Zarr (Analysis) | WebDataset (Training) |
| :---- | :---- | :---- | :---- |
| **Cloud Storage Friendly** | No (Monolithic) | Yes (Key-Value Chunks) | Yes (Sharded Tars) |
| **Parallel Write Support** | Poor (Global Lock) | Excellent (Chunk-level) | Excellent (Shard-level) |
| **Read Pattern** | Strided / Sliced | Random Access / Sliced | Sequential Streaming |
| **Throughput (GB/s)** | Limited by metadata | High | Line-rate (Max) |
| **Simplicity** | Complex Libraries | Pure Python/JSON | Standard Tar Tools |
| **Primary Use Case** | Local Archival | Interactive Analysis | Large-Scale ML Training |

## **5\. Documentation, Reproducibility, and Roadmap**

### **5.1 Jupyter Book as the Documentation Standard**

Section 7 implies a rigorous standard for documentation that supports the "Open" nature of the project. unified-qec should adopt **Jupyter Book** as its documentation engine. Unlike static site generators like Sphinx (upon which it is built), Jupyter Book treats executable notebooks as primary source files. This means the documentation *is* the test suite. A "Getting Started" guide in the documentation can be a notebook that actually pulls a model from the Model Zoo, streams data from the Open Data bucket, and runs a Sinter decoding task. This guarantees that the documentation never drifts from the code implementation.

### **5.2 Implementation Roadmap**

To achieve Section 7 compliance, the following phased execution is recommended:

1. **Phase I: Sinter Core (Weeks 1-4):** Develop the UnifiedQECDecoder class. Benchmark the compile\_decoder\_for\_dem implementation against file-based decoding to validate the expected 10x-50x speedup. Verify correctness against PyMatching v2.  
2. **Phase II: The Zoo (Weeks 5-8):** Implement the Orbax checkpointing wrapper. Establish the Hugging Face organization. Build the CI/CD pipelines that automatically push trained models to the Hub with auto-generated Model Cards.  
3. **Phase III: Data Scale-Up (Weeks 9-12):** Deploy the WebDataset generation pipeline. Generate a reference "Open Data" release (e.g., 1 billion shots of Rotated Surface Code d=11). Verify streaming performance on a multi-node JAX cluster.  
4. **Phase IV: Integration & Verification (Weeks 13-16):** Finalize the Jupyter Book documentation. Conduct a full "Section 7" audit, ensuring that a fresh user can clone the repo, download a model, and reproduce a SOTA benchmark plot using only the documented commands.

## **6\. Conclusion**

The "Section 7" requirements define a maturity model for QEC software. They demand a move away from the isolated, single-file scripts of the past toward a connected, cloud-native ecosystem. By integrating **Sinter**, unified-qec gains the engine needed for exascale simulation. By adopting **Orbax** and the **Hugging Face Hub**, it creates a living library of knowledge (The Model Zoo) rather than a static code dump. And by leveraging **WebDataset** and **Zarr**, it ensures that the data generated by this engine is accessible and usable by the global research community. Implementing this architecture will position the HoloJAX SDK as the foundational platform for the next decade of quantum fault tolerance research.

#### **Works cited**

1\. justinarndt/FB: FINAL BOSS: A Holo-Neural hybrid control ... \- GitHub, https://github.com/justinarndt/FB 2\. sinter \- PyPI, https://pypi.org/project/sinter/ 3\. Stim/doc/sinter\_command\_line.md at main · quantumlib/Stim \- GitHub, https://github.com/quantumlib/Stim/blob/main/doc/sinter\_command\_line.md 4\. Ultimate guide to huggingface\_hub library in Python \- Deepnote, https://deepnote.com/blog/ultimate-guide-to-huggingfacehub-library-in-python 5\. Checkpointing Flax NNX Models with Orbax (Part 2\) \- YouTube, https://www.youtube.com/watch?v=MJm5qbTdc-o 6\. A Comparison of HDF5, Zarr, and netCDF4 in Performing Common I/O Operations \- arXiv, https://arxiv.org/pdf/2207.09503 7\. \[D\] Best practices for storing multi-TB image datasets for use w/ PyTorch \- Reddit, https://www.reddit.com/r/MachineLearning/comments/1amu9ei/d\_best\_practices\_for\_storing\_multitb\_image/ 8\. Sinter v1.13 Python API Reference · quantumlib/Stim Wiki \- GitHub, https://github.com/quantumlib/Stim/wiki/Sinter-v1.13-Python-API-Reference 9\. Sinter v1.12 Python API Reference · quantumlib/Stim Wiki \- GitHub, https://github.com/quantumlib/Stim/wiki/Sinter-v1.12-Python-API-Reference 10\. abc — Abstract Base Classes — Python 3.14.3 documentation, https://docs.python.org/3/library/abc.html 11\. Integrating an arbitrary decoder into the stim framework, https://quantumcomputing.stackexchange.com/questions/28837/integrating-an-arbitrary-decoder-into-the-stim-framework 12\. Releases · quantumlib/Stim \- GitHub, https://github.com/quantumlib/Stim/releases 13\. Implement \`compile\_decoder\_for\_dem\` for best performance from sinter · Issue \#32 · quantumgizmos/ldpc \- GitHub, https://github.com/quantumgizmos/ldpc/issues/32 14\. numpy.unpackbits() \- JAX documentation, https://docs.jax.dev/en/latest/\_autosummary/jax.numpy.unpackbits.html 15\. Sparse Blossom: correcting a million errors per core second with minimum-weight matching, https://quantum-journal.org/papers/q-2025-01-20-1600/ 16\. PyMatching \- Release 2.1.dev1 Oscar Higgott and Craig Gidney, https://pymatching.readthedocs.io/\_/downloads/en/latest/pdf/ 17\. oscarhiggott/PyMatching: PyMatching: A Python/C++ library for decoding quantum error correcting codes with minimum-weight perfect matching. \- GitHub, https://github.com/oscarhiggott/PyMatching 18\. yuewuo/fusion-blossom: A fast minimum-weight perfect matching solver for quantum error correction \- GitHub, https://github.com/yuewuo/fusion-blossom 19\. fusion-blossom 0.1.3 \- PyPI, https://pypi.org/project/fusion-blossom/0.1.3/ 20\. Description of the internal decoders of" sinter" \- Quantum Computing Stack Exchange, https://quantumcomputing.stackexchange.com/questions/29761/description-of-the-internal-decoders-of-sinter 21\. Migrate checkpointing to Orbax \- Flax \- Read the Docs, https://flax.readthedocs.io/en/v0.6.10/guides/orbax\_upgrade\_guide.html 22\. Using Orbax to checkpoint flax \`TrainState\` with new \`CheckpointManager\` API, https://stackoverflow.com/questions/78033458/using-orbax-to-checkpoint-flax-trainstate-with-new-checkpointmanager-api 23\. CheckpointManager — Orbax documentation \- Read the Docs, https://orbax.readthedocs.io/en/latest/api\_reference/checkpoint.checkpoint\_manager.html 24\. How to restore a orbax checkpoint with jax/flax? \- Stack Overflow, https://stackoverflow.com/questions/78376465/how-to-restore-a-orbax-checkpoint-with-jax-flax 25\. Checkpointing with Orbax, https://orbax.readthedocs.io/en/latest/guides/checkpoint/orbax\_checkpoint\_101.html 26\. Uploading models \- Hugging Face, https://huggingface.co/docs/hub/en/models-uploading 27\. Upload files to the Hub \- Hugging Face, https://huggingface.co/docs/huggingface\_hub/guides/upload 28\. huggingface/huggingface\_hub: The official Python client for the Hugging Face Hub. \- GitHub, https://github.com/huggingface/huggingface\_hub 29\. Understand caching \- Hugging Face, https://huggingface.co/docs/huggingface\_hub/guides/manage-cache 30\. Manage huggingface\_hub cache-system, https://huggingface.co/docs/huggingface\_hub/v0.25.0/en/guides/manage-cache 31\. Quickstart \- Hugging Face, https://huggingface.co/docs/huggingface\_hub/quick-start 32\. THE LANDSCAPE OF ML DOCUMENTATION TOOLS \- Hugging Face, https://huggingface.co/docs/hub/en/model-card-landscape-analysis 33\. Cloud-Performant NetCDF4/HDF5 Reading with the Zarr Library | by Richard Signell, https://medium.com/pangeo/cloud-performant-reading-of-netcdf4-hdf5-data-using-the-zarr-library-1a95c5c92314 34\. Efficient PyTorch I/O library for Large Datasets, Many Files, Many GPUs, https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/ 35\. Why I Chose WebDataset for Training on 50TB of Data? | by Ahmad Sachal | Red Buffer, https://medium.com/red-buffer/why-did-i-choose-webdataset-for-training-on-50tb-of-data-98a563a916bf 36\. Optimizing Data Loading Performance in JAX with jax-dataloader and Grain \- Medium, https://medium.com/google-developer-experts/optimizing-data-loading-performance-in-jax-with-jax-dataloader-and-grain-75adcd10f614 37\. \[Question\] Comparison with the zarr format? · Issue \#527 · huggingface/safetensors \- GitHub, https://github.com/huggingface/safetensors/issues/527 38\. How Jupyter Book and Sphinx relate to one another \- Read the Docs, https://jupyter-book.readthedocs.io/v1/explain/sphinx.html 39\. Sphinx usage and customization \- Jupyter Book, https://jupyterbook.org/en/latest/sphinx/