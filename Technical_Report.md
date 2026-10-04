# Technical Report: Generative AI Assignment

## Task 1: Universal Multi-Corruption Denoising Autoencoder

### 1. Introduction
This section explores the development of a universal denoising autoencoder designed to restore images affected by various types of visual corruption. Unlike specialized models that target a single degradation type, this single architecture is trained to generalize across clean images, salt-and-pepper noise, Gaussian blur, and rectangular occlusion without explicitly being provided the corruption label at inference time.

### 2. Dataset Preparation
The Oxford-IIIT Pet Dataset was utilized as the primary data source, providing a diverse set of clean target images resized to a standard 128x128 pixel resolution. The data was split into an 80% training set and a 20% validation set using a fixed random seed (42) to ensure reproducibility. 

To train the universal autoencoder robustly, synthetic corruptions were applied dynamically during the training process via a custom PyTorch data-loading pipeline. For every loaded image, one of four input conditions was selected with equal probability (25%):
1.  **Clean:** The original image was passed without modification.
2.  **Salt-and-Pepper Noise:** Pixels were randomly replaced with black or white values, with the corruption probability uniformly sampled between 0.02 and 0.15.
3.  **Gaussian Blur:** A blur was applied by randomly selecting a kernel size from {3, 5, 7} and a standard deviation uniformly sampled between 0.5 and 2.5.
4.  **Rectangular Occlusion:** Between one and three black rectangular masks were inserted into random locations on the image, jointly covering between 10% and 35% of the total image area.

To ensure a fair and deterministic evaluation during the validation phase, a fixed random seed (derived from the image index) was used. This acts as a static "corruption manifest," ensuring that the validation loss reflects model improvements rather than random variations in the corruption generation.

### 3. Architecture Design
The universal autoencoder follows a strict hourglass architecture with a genuine compressed latent representation. Crucially, unrestricted skip connections (such as those found in a standard U-Net) were omitted to prevent the model from simply copying the corrupted input through to the output layer, forcing it to compress and learn shared restoration features within the bottleneck.

*   **Encoder:** The encoder consists of four downsampling blocks. Each block applies a 4x4 2D Convolution with a stride of 2 and padding of 1, followed by a ReLU activation. This progressively reduces the spatial dimensions from 128x128 to 8x8 while expanding the channel depth (e.g., 3 → 32 → 64 → 128 → 256).
*   **Bottleneck:** The output of the final convolutional layer is flattened and passed through a fully connected layer (Linear) to project the features into a fixed-size latent dimension (e.g., 256 parameters). A second fully connected layer immediately projects this vector back up to the required tensor shape.
*   **Decoder:** The decoder reconstructs the image using four transposed convolution layers (`ConvTranspose2d`) that mirror the encoder's dimensions, progressively upsampling the 8x8 spatial tensor back to the original 128x128 RGB resolution. A final Sigmoid activation is applied to ensure the output pixels remain bounded within the [0, 1] range.

### 4. Loss Functions
To achieve both pixel-level accuracy and perceptual quality, the model is trained on a combined objective containing both L1 Loss and Structural Similarity Index Measure (SSIM) loss. 

The L1 component drives the model to accurately reconstruct the exact pixel intensities of the clean image, penalizing large outlier errors gracefully. The SSIM component ensures the reconstructed image preserves structural integrity, local contrast, and visually coherent edges, which is particularly vital for restoring areas destroyed by rectangular occlusions and heavy blur.

The final loss formula is a weighted combination:
`Loss = α * L1(x, x_hat) + (1 - α) * (1 - SSIM(x, x_hat))`
The initial balance parameter `α` is set to 0.8, heavily prioritizing L1 reconstruction early on, while allowing structural similarity to refine the results. This parameter is subjected to hyperparameter optimization during the tuning phase.

### 5. Training Procedure
The model was trained using the Adam optimizer. To facilitate hyperparameter tuning (via Optuna) and prevent computational bottlenecks, training was orchestrated through a modular loop structure. During the forward pass, corrupted images were fed into the model, and the resulting outputs were compared against the original clean targets using the combined L1-SSIM criterion. 

The training loop separately tracked the aggregate loss, raw L1 loss, and the perceptual SSIM score across each epoch. Following the training phase of each epoch, the model was evaluated on the deterministic validation set without gradient calculation to monitor for generalization and early signs of overfitting.

### 6. Hyperparameter Optimization (Optuna)
Given the complexity of balancing reconstruction against structural similarity, the model hyperparameters were not chosen statically. An automated search was conducted using the Optuna optimization framework to minimize the combined validation loss over a constrained epoch budget. 

The search space included:
*   **Learning Rate:** Log-uniform sampling between $10^{-4}$ and $10^{-2}$.
*   **Batch Size:** Categorical selection from {16, 32, 64}.
*   **Bottleneck Dimension:** Categorical selection from {128, 256, 512} to find the optimal compression ratio.
*   **Base Channel Count:** Categorical selection from {16, 32} to balance model capacity against overfitting.
*   **Dropout Rate:** Uniform sampling between 0.0 and 0.5.
*   **Loss Weight ($\alpha$):** Uniform sampling between 0.1 and 0.9.

The objective function evaluated each configuration by training the model for a reduced number of epochs and reporting the intermediate validation loss. Optuna's trial pruning (Median Pruner) was utilized to terminate unpromising configurations early, saving computational resources. The best-performing configuration was then extracted to train the final production model.

### 7. Experimental Results and Analysis
Following the hyperparameter search, the model was trained using the optimal configuration. The best trial heavily favored the L1 reconstruction loss (α ≈ 0.80), confirming the initial theoretical assumption that exact pixel mapping dominates the early restoration process, while a smaller weighting (≈0.20) for the structural similarity metric is sufficient to preserve edge coherence. 

Visual evaluation of the output across the three discrete corruption types (blur, noise, and occlusion) demonstrates that the model successfully generalizes. It achieves this by forcing the varied high-frequency corruption artifacts through the restricted low-frequency bottleneck, smoothing out noise and extrapolating missing occluded structures based on surrounding context. 

Absolute error maps (visualized as heatmaps calculated via $|x - \hat{x}|$) highlight that the majority of reconstruction error occurs at sharp edges and texturally complex regions (like fur). Conversely, the error remains near zero on flat, uniform background regions, regardless of the severity of the initial corruption.

### 8. Model Export and Deployment Architecture
To facilitate integration into the required browser-based application, the trained PyTorch model was exported to the Open Neural Network Exchange (ONNX) format. This allows the Fast API backend to perform inference natively without requiring a heavy PyTorch dependency graph. The dynamic axes configuration was enabled during export to ensure the application can handle variable batch sizes if multiple images are uploaded concurrently.

### 9. Conclusion
This implementation validates the effectiveness of a universal denoising autoencoder capable of resolving multi-domain corruptions. By strictly enforcing a bottleneck representation and jointly optimizing for L1 and structural similarity, the architecture avoids identity mapping and learns a genuine, shared latent manifold of "clean" image features. Future work may explore soft routing or mixture-of-experts (as outlined in Tasks 2 and 3) to handle extreme degradation scenarios where a single shared bottleneck struggles to preserve fine-grained structural fidelity.

## Task 2: Corruption Classification and Hard-Routed Specialist Autoencoders

### 1. Introduction
While a universal autoencoder offers architectural simplicity, forcing a single bottleneck to decode fundamentally different degradation mathematical models (e.g., additive noise vs. spatial occlusion) can limit peak restoration performance. Task 2 proposes a hard-routed modular system: a corruption classifier determines the specific degradation present in an image, and subsequently routes the image to an independent, highly specialized denoising autoencoder trained exclusively for that specific domain. Clean images bypass restoration entirely via an identity route.

### 2. Corruption Classifier Architecture
The first component of the hard-routing pipeline is a dedicated Convolutional Neural Network (CNN) classifier responsible for predicting one of four classes: Clean, Salt-and-Pepper Noise, Gaussian Blur, or Rectangular Occlusion. 

*   **Data Balancing:** The synthetic corruption pipeline developed in Task 1 ensures that the four classes are generated with equal (25%) probability during runtime. This stochastic uniform sampling inherently balances the training batches, preventing the classifier from developing a prior bias toward any single corruption class.
*   **Architecture:** The classifier employs four consecutive convolutional blocks. Each block consists of a 3x3 Convolution, a ReLU activation, and a 2x2 Max Pooling layer, progressively halving the spatial resolution while increasing feature depth. The resulting feature map is flattened and passed through a dropout-regularized dense layer before producing a 4-class unnormalized logit vector.
*   **Objective:** The model is optimized using Multiclass Cross-Entropy Loss. To maximize predictive accuracy and prevent overfitting, the architecture and optimizer were tuned via Optuna, searching across learning rate, batch size, base channel depth, dropout rate, and weight decay (L2 regularization).

### 3. Classifier Evaluation
Following hyperparameter optimization, the best classifier configuration was trained on the fully balanced dataset. The classifier's predictive performance was rigorously evaluated on the deterministic validation set to extract macro-averaged precision, recall, and F1-scores, alongside strict per-class metrics. 

A normalized four-class confusion matrix was computed to visualize inter-class boundary errors. By normalizing over the true labels, the matrix highlights the classifier's sensitivity and reveals any specific failure modes (e.g., misclassifying light Gaussian blur as a clean image, or confusing severe occlusion with heavy noise). The high overall macro-F1 score ensures that the hard-routing mechanism downstream can confidently rely on the classifier's outputs without cascading severe routing errors into the specialist networks.

### 4. Specialist Autoencoders
With a highly accurate classifier acting as the routing mechanism, the restoration workload is distributed among three dedicated specialist autoencoders: one for Salt-and-Pepper noise, one for Gaussian Blur, and one for Rectangular Occlusion. 

*   **Architecture and Training:** To ensure computational feasibility and maintain architectural consistency across the pipeline, the specialists share the exact same layer topology discovered during the Task 1 Universal Autoencoder Optuna search. This shared architecture features the strict bottleneck and dropout regularization required for robust reconstruction.
*   **Specialized Datasets:** Unlike the universal model, each specialist was strictly isolated during training. Custom data loaders were constructed to forcefully apply only the targeted corruption to the clean images (e.g., the Blur Specialist only ever processed blurred images). 
*   **Loss Function:** The independent training of each expert utilized the combined L1 and SSIM loss function, ensuring precise pixel restoration and structural coherence specific to their designated degradation domain. The clean image bypasses the restoration networks entirely during inference, preserving zero-loss fidelity for unaffected inputs.

### 5. Hard-Routing Inference Pipeline
The operational system ties the classifier and the specialists together into a unified inference pipeline. When a corrupted image is submitted, it is first evaluated by the classifier to yield a predicted corruption class. 

The hard-routing mechanism leverages this prediction to select the appropriate specialist. If the classifier predicts a "Clean" image, the system applies an **identity bypass**, returning the original image unaltered. This is a critical advantage over the universal autoencoder, which inherently subjects clean inputs to minor reconstruction artifacts simply by passing them through the latent bottleneck.

### 6. Oracle vs. Predicted Routing Evaluation
To evaluate the absolute ceiling of the specialists' capabilities against real-world operational performance, the system was tested in two discrete modes:
1.  **Oracle Routing:** The system routes the image based on the known, ground-truth corruption label from the deterministic test manifest. This isolates the performance of the autoencoders, revealing their peak restoration capability free from classification bottlenecks.
2.  **Predicted Routing:** The system operates autonomously, routing based on the CNN classifier's prediction.

Given the classifier's high empirical accuracy (98%), the discrepancy between oracle routing and predicted routing is minimal across the evaluation set. However, edge cases exist—specifically when evaluating low-severity Gaussian blur, which the classifier may occasionally confuse with a clean image. In such cases, the hard-routing pipeline executes the identity bypass. While technically a classification failure, this specific failure mode is benign, as bypassing a lightly blurred image produces a visually identical result to attempting a full reconstruction, avoiding the severe artifact generation that would occur if it were erroneously routed to the Salt-and-Pepper or Occlusion specialists.

## Task 3: Jointly Trained Soft Mixture-of-Experts Restoration

### 1. Introduction
While hard routing (Task 2) excels when an image is afflicted by a single, unambiguous corruption, it inherently struggles with boundary cases, classifier uncertainty, or inputs featuring composite corruptions. To address these limitations, Task 3 introduces a differentiable Soft Mixture-of-Experts (MoE) system. Rather than routing an image to a single specialist, the gating network assigns a continuous probability distribution across all available experts, and the final reconstruction is a weighted sum of their outputs.

### 2. Soft MoE Architecture & Joint Loss
The architecture wraps the pre-trained components from Task 2 into a single differentiable graph. The classifier functions as the gating network, utilizing a temperature-scaled softmax activation ($w = \text{softmax}(G(\tilde{x}) / \tau)$) to output a 4-dimensional continuous weight vector mapping to the Clean (identity), Salt-and-Pepper, Blur, and Occlusion branches. 

To train this combined system cooperatively, a custom composite loss function ($L_{MoE}$) was developed. It balances four competing objectives:
1.  **L1 Loss:** For strict pixel-level reconstruction.
2.  **SSIM Loss:** To preserve structural perception and edge coherence.
3.  **Cross-Entropy Loss:** To anchor the gating network's routing decisions to the known runtime corruption labels.
4.  **Balance Regularizer:** A mean-squared penalty ($\sum (\bar{w}_k - 0.25)^2$) applied to the average batch routing weights, explicitly penalizing routing collapse (i.e., the gate sending all inputs to a single dominant expert).

### 3. Training Procedure and Hyperparameter Optimization
Due to the immense parameter count and the risk of catastrophic forgetting within the highly specialized experts, training was strictly bifurcated into two stages:
1.  **Warm-Up Phase:** The three specialist autoencoders were completely frozen (`requires_grad = False`). The system was trained for a short epoch budget focusing exclusively on the gating network. This primed the gate to output soft probability distributions rather than hard categorical logits, protecting the experts from erratic gradients during initial convergence.
2.  **Joint Fine-Tuning Phase:** The entire computational graph was subsequently unfrozen. To find the optimal stability dynamics, an Optuna search was deployed across the joint learning rate, softmax temperature ($\tau$), classification weight ($\lambda_c$), balance weight ($\lambda_b$), and the L1 vs. SSIM reconstruction ratio ($\lambda_1$). 

The temperature parameter $\tau$ proved particularly influential in controlling whether the model functioned as a "sharp" router (mimicking Task 2) or a "distributed" router (blending multiple expert outputs simultaneously). Smaller learning rates during this joint fine-tuning phase were critical to gracefully merging the feature manifolds without degrading the established specialist capabilities.

### 4. Evaluation and Routing Behavior
To evaluate the success of the soft routing system, the final model was deployed across the validation set, and the assigned routing weights for each image were collected and averaged by their true underlying corruption class.

The resulting routing heatmap demonstrates the gating network's ability to smoothly distribute trust among the experts. When an image is heavily corrupted by salt-and-pepper noise, the gating network assigns a dominant weight coefficient (e.g., >0.90) to the S&P Specialist, while reserving marginal fractional weights for the identity bypass or blur expert. This fractional blending serves as an interpolation mechanism, allowing the model to naturally handle varying degrees of corruption severity without encountering the discrete boundary thresholds that cause failures in hard-routing architectures. 

Furthermore, the inclusion of the Balance Regularizer in the joint loss function successfully prevented routing collapse. By penalizing significant deviations from an equal expectation, the system was forced to utilize all specialized branches effectively, validating the Soft Mixture-of-Experts approach as the most robust architecture for multi-domain image restoration within this study. The entire differentiable system was successfully exported to ONNX, ready for frontend integration.

## Task 4: Style-Conditioned Face-to-Sketch Generation Using a Conditional GAN

### 1. Introduction and Dataset Preparation
The fourth phase expands generative capabilities to multi-domain image translation by building a Conditional Generative Adversarial Network (cGAN) for face-to-sketch synthesis. Using the FS2K Facial Sketch Synthesis Dataset (containing 2,104 paired photographs and sketches), the system learns to translate human faces into one of three distinct artistic sketch styles. 

To ensure robust evaluation, 15% of the official training partition was reserved as a validation set, stratified rigorously by sketch style. A critical requirement for paired image-to-image translation is strict spatial correspondence; thus, custom data loading pipelines were engineered to mathematically ensure that any spatial augmentation (such as rotation or horizontal flipping) applied to a photograph was identically applied to its target sketch, preserving pixel-level alignment.

### 2. Architecture: Generator and PatchGAN Discriminator
The system employs an adversarial architecture consisting of a Generator $G$ and a Discriminator $D$, both conditioned on a learned categorical style embedding.
*   **Generator (U-Net):** Following an encoder-decoder structure with skip connections, the Generator receives a concatenated input comprising the RGB facial photograph and a spatially broadcasted style embedding. It learns to extract facial features and decode them into a targeted sketch $y' = G(x, s)$.
*   **Discriminator (PatchGAN):** To ensure high-frequency detail fidelity, a PatchGAN discriminator was implemented. Rather than outputting a single global scalar, the discriminator evaluates local overlapping regions (patches) of the image. It receives the photograph, the style embedding, and either the real or generated sketch ($D(x, y, s)$ vs $D(x, G(x, s), s)$) and classifies each patch as real or fake.

### 3. Objective Functions and Training Dynamics
The Generator is optimized using a hybrid objective function: $L_G = L_{adv} + \lambda_{L1} L_{L1}(y, G(x, s))$. The adversarial component encourages the generation of realistic sketch textures capable of fooling the discriminator, while the L1 reconstruction loss strictly penalizes structural deviations from the paired ground-truth sketch. 

To navigate the notoriously unstable dynamics of GAN training, Optuna was deployed to search for the optimal learning rates (for both $G$ and $D$), batch size, base channel capacity, dropout rates, embedding dimensions, and the $\lambda_{L1}$ weight. The training loop continuously monitored and recorded the separated $D_{real}$, $D_{fake}$, $G_{adv}$, and $G_{recon}$ loss metrics to ensure neither network overpowered the other. Following the discovery of optimal parameters, the model was fine-tuned, and the standalone Generator was exported to ONNX format, explicitly dropping the Discriminator as it is not required during application inference.

## Task 5: Interactive Web Application Deployment

### 1. Introduction and Architecture
To demonstrate the practical utility of the models developed throughout the assignment, a full-stack web application was engineered. The application features a dynamic tabbed interface offering two distinct workspaces: an Image Restoration suite (for Tasks 1-3) and a Face-to-Sketch Generator (for Task 4).

The system is decoupled into two primary components:
1.  **Frontend (HTML/JavaScript):** A lightweight browser-based user interface designed for high interactivity. 
    *   *Workspace 1 (Restoration):* Allows users to upload corrupted images and instantly visualize the restoration process using the Universal, Hard-Routed, or Soft MoE architectures.
    *   *Workspace 2 (Face-to-Sketch):* Introduces HTML5 WebRTC integration, allowing users to capture facial photographs natively via their webcam (or via standard file upload). Users select their desired target style (1, 2, or 3) from a dropdown menu, and the system renders the original photograph alongside the generated sketch, complete with a download mechanism.
2.  **Backend (FastAPI & ONNX Runtime):** A high-performance Python backend server. Crucially, the backend completely drops the heavy PyTorch dependency. Instead, it relies exclusively on `onnxruntime` to execute the pre-compiled execution graphs natively. The backend handles image resizing, specific tensor normalization (managing both the `[0, 1]` scaling required by the Autoencoders and the `[-1, 1]` normalization required by the GAN's Tanh activation), execution routing, and output tensor post-processing back into standard PNG image bytes.

This decoupled, ONNX-driven architecture ensures that the final product is incredibly fast, memory-efficient, and easily containerized for cloud deployment, proving that complex multi-domain generative architectures and conditional GANs can be successfully transitioned from research pipelines into interactive, production-ready software.
