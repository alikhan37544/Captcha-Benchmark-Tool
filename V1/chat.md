### Key Models for Captcha Solving
Research suggests several open-source models on Hugging Face excel at vision-based tasks like captcha solving through OCR or image-to-text extraction. Evidence leans toward lightweight vision-language models (VLMs) and dedicated OCR models for fast, local deployment, though performance can vary with captcha complexity—such as distorted text or backgrounds. It's likely that combining models or fine-tuning may improve accuracy for specific captcha types, acknowledging that no model is foolproof due to evolving captcha designs.

- **Top Lightweight Picks**: Models like nanoVLM (222M parameters) and SmolVLM (2B) offer quick responses on your hardware, balancing speed and capability.
- **Best for OCR-Focused Captchas**: Dedicated models like GLM-OCR or TroCR variants seem effective, with high accuracy on printed or distorted text.
- **Versatile VLMs**: Qwen2.5-VL (7B) and MiniCPM-V (8B) provide strong multimodal reasoning, suitable for prompting to extract captcha text.
- **Hardware Fit**: All listed models can run locally on your RTX 4060 Ti 16GB (assuming typo in "5060") with quantization (e.g., via GGUF formats in LM Studio) for faster inference.

#### Deployment Considerations
These models integrate well with LM Studio, which pulls from Hugging Face. Use quantized versions (e.g., 4-bit or 8-bit) to fit within 16GB VRAM and 64GB RAM for instant responses—aim for models under 8B parameters unquantized. For base64 images, decode them in your script (e.g., Python with base64 and PIL) before passing to the model. Prompt VLMs like: "Extract the exact text from this captcha image."

#### Potential Challenges
Captchas often include noise, so models may need prompting tweaks or preprocessing (e.g., image enhancement). Community feedback highlights that while these solve many standard captchas, advanced ones (e.g., reCAPTCHA) might require additional techniques, respecting ethical use.

---

### Comprehensive Survey of Open-Source Vision Models for Captcha Solving
This detailed overview compiles the most relevant open-source models available on Hugging Face as of early 2026, based on trending, downloads, and benchmarks for vision capabilities, particularly OCR and image-to-text extraction suitable for captcha solving. These models can process base64-encoded images (after decoding) and output text responses, making them ideal for building agents. I've prioritized lightweight options for speed on your hardware (RTX 4060 Ti 16GB VRAM, 64GB RAM), but included a broad range up to ~11B parameters, as quantization allows larger models to run efficiently via LM Studio. The list draws from recent benchmarks, community discussions, and model repositories, focusing on those with proven or potential captcha-solving efficacy.

Models are categorized for clarity: dedicated OCR for precise text extraction from distorted images, and VLMs for flexible prompting (e.g., "What text is in this captcha?"). All are open-source, downloadable from Hugging Face, and deployable locally. For speed, prefer models under 4B parameters; they can achieve sub-second inference with quantization. I've noted sizes, key features, and why they're suitable, including any captcha-specific mentions from sources.

#### Dedicated OCR Models
These are specialized for optical character recognition, excelling at extracting text from captcha-like images with distortions, noise, or varying fonts. They're typically lighter and faster than VLMs for pure text extraction.

- **GLM-OCR (zai-org/GLM-OCR)**: ~1B parameters. A lightweight OCR model optimized for text extraction from images, including multilingual support. Benchmarks show strong performance on OCR tasks, making it ideal for captchas with printed text. URL: https://huggingface.co/zai-org/GLM-OCR
- **ocr-captcha-v3 (anuashok/ocr-captcha-v3)**: ~0.3B parameters. Fine-tuned TroCR-base for specific captcha types with wavy or noisy text. Achieves low character error rates (CER ~0.02) on custom datasets, perfect for direct captcha solving. URL: https://huggingface.co/anuashok/ocr-captcha-v3
- **ocr-for-captcha (keras-io/ocr-for-captcha)**: ~0.3B parameters (TF-Keras based). Designed explicitly for reading captchas using CNN-RNN architecture. Includes notebooks for local training and inference, with high accuracy on synthetic captchas. URL: https://huggingface.co/keras-io/ocr-for-captcha
- **manga-ocr-base (kha-white/manga-ocr-base)**: ~0.1B parameters. Lightweight for scene text in images, effective on distorted or artistic text similar to captchas. Fast inference, supports Japanese but adaptable. URL: https://huggingface.co/kha-white/manga-ocr-base
- **nemotron-ocr-v1 (nvidia/nemotron-ocr-v1)**: ~0.5B parameters. NVIDIA's OCR model for high-accuracy text extraction, including handwriting. Suitable for captcha variants with mixed styles. URL: https://huggingface.co/nvidia/nemotron-ocr-v1
- **PaddleOCR-VL (PaddlePaddle/PaddleOCR-VL)**: ~1B parameters. Multimodal OCR with layout analysis, excels at structured text in images. Benchmarks average ~79% on OCR tasks, good for complex captchas. URL: https://huggingface.co/PaddlePaddle/PaddleOCR-VL
- **OlmOCR-7B (allenai/olmOCR-7B-0825)**: ~7B parameters. VLM-based OCR fine-tune of Qwen2.5-VL, supports markdown output and grounding. High scores (~85%) on benchmarks, versatile for captcha text with context. URL: https://huggingface.co/allenai/olmOCR-7B-0825
- **dots.ocr (rednote-hilab/dots.ocr)**: ~0.5B parameters. Small, efficient OCR VLM highlighted as top open-source for 2025-2026. Excels at precise text detection in noisy images. URL: https://huggingface.co/rednote-hilab/dots.ocr
- **Nanonets-OCR2-3B (nanonets/nanonets-OCR2-3B)**: ~3B parameters. Supports multiple formats like markdown/HTML, with grounding. Average ~82% on benchmarks, good for captcha with layouts. URL: https://huggingface.co/nanonets/nanonets-OCR2-3B
- **Surya (vikhyatk/surya)**: ~0.5B parameters. Layout-focused OCR for documents, but adaptable to captchas. Strong on structured text extraction. URL: https://huggingface.co/vikhyatk/surya

#### Vision-Language Models (VLMs)
These multimodal models handle images and text prompts, allowing you to query "Solve this captcha by reading the text." They're more flexible but may require careful prompting for accuracy. Lightweight ones prioritize speed.

- **nanoVLM (huggingface/nanoVLM)**: ~222M parameters. Extremely lightweight for edge devices, processes images for text generation. Ideal for fast, local captcha agents with minimal RAM use. URL: https://huggingface.co/huggingface/nanoVLM
- **SmolVLM-Instruct (HuggingFaceTB/SmolVLM-Instruct)**: ~2B parameters. Small VLM for vision tasks, SOTA for its size in memory efficiency. Runs quickly on consumer GPUs, good for interactive captcha solving. URL: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
- **Qwen2.5-VL-7B (Qwen/Qwen2.5-VL-7B-Instruct)**: ~7B parameters. Top performer in OCR benchmarks (~75% accuracy), multilingual, handles high-res images. Quantized versions fit your VRAM for fast responses. URL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- **MiniCPM-V-2.6 (openbmb/MiniCPM-V-2_6)**: ~8B parameters. Leads OCRBench with support for 30+ languages and high-res images. Efficient for local deployment, great for diverse captchas. URL: https://huggingface.co/openbmb/MiniCPM-V-2_6
- **Llama-3.2-Vision-11B (meta-llama/Llama-3.2-11B-Vision-Instruct)**: ~11B parameters. Strong in VQA and OCR, fits quantized on 16GB VRAM. Community notes high effectiveness for text extraction. URL: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- **Moondream2 (vikhyatk/moondream2)**: ~2B parameters. Compact VLM for image understanding, fine-tuned for OCR-like tasks. Very fast, suitable for real-time captcha agents. URL: https://huggingface.co/vikhyatk/moondream2
- **Idefics2 (HuggingFaceM4/idefics2-8b)**: ~8B parameters. Mistral-based VLM with strong multimodal reasoning. Efficient for local runs, handles captcha text via prompts. URL: https://huggingface.co/HuggingFaceM4/idefics2-8b
- **Llava-Next-Interleave-Qwen-7B (lmms-lab/llava-next-interleave-qwen-7b)**: ~7B parameters. Supports interleaved images, excellent for OCR. Balanced for speed and accuracy on your setup. URL: https://huggingface.co/lmms-lab/llava-next-interleave-qwen-7b
- **PaliGemma-3B (google/paligemma-3b-mix-224)**: ~3B parameters. Hybrid vision-language for captioning and extraction. Lightweight and promptable for captchas. URL: https://huggingface.co/google/paligemma-3b-mix-224
- **LFM2-VL-450M (LiquidAI/LFM2-VL-450M)**: ~0.45B parameters. Hyper-efficient for on-device, low-latency vision tasks. Perfect for speedy captcha solving with minimal resources. URL: https://huggingface.co/LiquidAI/LFM2-VL-450M

#### Benchmark Comparison Table
Here's a table summarizing key models' performance on relevant benchmarks (e.g., OCRBench, OlmOCR), sizes, and speed estimates (inference time on similar hardware, approximate with quantization). Data aggregated from sources; higher scores indicate better text extraction accuracy.

| Model Name | Parameters | OCRBench Score | OlmOCR Avg Score | Est. Inference Time (sec/image) | Best For Captchas? | Source Notes |
|------------|------------|----------------|------------------|---------------------------------|---------------------|--------------|
| GLM-OCR | 1B | N/A | ~80% | <0.5 | Printed text | Trending on HF, benchmarks highlight efficiency. |
| ocr-captcha-v3 | 0.3B | N/A | N/A | <0.3 | Distorted captchas | Fine-tuned for captchas, low CER. |
| Qwen2.5-VL-7B | 7B | ~75% | ~75% | 1-2 | Multilingual, complex | Tops open-source OCR lists. |
| MiniCPM-V-2.6 | 8B | ~85% (leader) | N/A | 1-3 | High-res captchas | Leads OCRBench. |
| SmolVLM-Instruct | 2B | ~70% | N/A | <1 | Fast local agents | SOTA for small size. |
| PaddleOCR-VL | 1B | N/A | ~79% | 0.5-1 | Structured/noisy | Enterprise-level OCR. |
| nanoVLM | 0.22B | N/A | N/A | <0.5 | Edge-speed | Tiny for instant responses. |
| Llama-3.2-Vision-11B | 11B | ~72% | N/A | 2-4 | Versatile prompting | Effective per community. |
| dots.ocr | 0.5B | N/A | High (top open) | <0.5 | Noisy images | 2025-2026 standout. |
| Moondream2 | 2B | ~68% | N/A | <1 | Quick OCR | Compact and fast. |

#### Additional Notes on Usage and Benchmarks
For your benchmark tool, test on datasets like CAPTURE (61K+ captchas from 31 vendors). VLMs like Qwen2.5-VL outperform on controversial benchmarks, but dedicated OCR like GLM-OCR is faster for simple text. Community suggests fine-tuning on captcha datasets for better results. Always verify ethically—models aren't for bypassing security.

### Key Citations
- [FineVision: Open Data is All You Need - Hugging Face](https://huggingface.co/spaces/HuggingFaceM4/FineVision)
- [keras-io/ocr-for-captcha - Hugging Face](https://huggingface.co/keras-io/ocr-for-captcha)
- [Are there any Open source/self hosted captcha solvers? - Reddit](https://www.reddit.com/r/webscraping/comments/1h1umpm/are_there_any_open_sourceself_hosted_captcha)
- [A Benchmark and Evaluation for LVLMs in CAPTCHA Resolving](https://arxiv.org/html/2512.11323v1)
- [anuashok/ocr-captcha-v3 - Hugging Face](https://huggingface.co/anuashok/ocr-captcha-v3)
- [Choosing the Right AI for Video Security: YOLO vs. Hugging Face vs ...](https://usa.seco.com/news/details/choosing-the-right-ai-for-video-security-yolo-vs-hugging-face-vs-mistral)
- [Hugging Face Pre-trained Models: Find the Best One for Your Task](https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best)
- [Supercharge your OCR Pipelines with Open Models - Hugging Face](https://huggingface.co/blog/ocr-open-models)
- [Best Local Vision-Language Models for Offline AI - Roboflow Blog](https://blog.roboflow.com/local-vision-language-models)
- [Mastering Vision Transformers with Hugging Face - Rapid Innovation](https://www.rapidinnovation.io/post/integrating-hugging-face-transformers-into-your-computer-vision-projects)
- [Image-to-Text Models - Hugging Face](https://huggingface.co/models?pipeline_tag=image-to-text)
- [Recommendations for Models that Handle Text and Screenshots for ...](https://discuss.huggingface.co/t/title-recommendations-for-models-that-handle-text-and-screenshots-for-qa/114665)
- [Top 10 most popular LLM models on Hugging Face - Cloudsmith](https://cloudsmith.com/blog/top-10-most-popular-llm-models-on-hugging-face)
- [Best small vision LLM for OCR? : r/LocalLLaMA - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1f71k60/best_small_vision_llm_for_ocr)
- [keras-io/ocr-for-captcha - Hugging Face](https://huggingface.co/keras-io/ocr-for-captcha)
- [Building Image-to-Text Matching System Using Hugging Face Open ...](https://youssefh.substack.com/p/building-image-to-text-matching-system)
- [Image-to-Text Models - a merve Collection - Hugging Face](https://huggingface.co/collections/merve/image-to-text-models)
- [Choosing the Right AI for Video Security: YOLO vs. Hugging Face vs ...](https://usa.seco.com/news/details/choosing-the-right-ai-for-video-security-yolo-vs-hugging-face-vs-mistral)
- [Multimodal Tasks and Models - Hugging Face Community Computer ...](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/tasks-models-part1)
- [10 Hugging Face Model Types and Domains that are Perfect for ...](https://openmetal.io/resources/blog/10-hugging-face-models-on-private-ai-infrastructure)
- [Vision Language Models (Better, faster, stronger) - Hugging Face](https://huggingface.co/blog/vlms-2025)
- [Best way to deploy a SLM/LLM model. Best library and approach?](https://discuss.huggingface.co/t/best-way-to-deploy-a-slm-llm-model-best-library-and-approach/138523)
- [SmolVLM - small yet mighty Vision Language Model - Hugging Face](https://huggingface.co/blog/smolvlm)
- [A Lightweight Multimodal Vision-Language Model by Hugging Face](https://medium.com/@uzbrainai/exploring-nanovlm-a-lightweight-multimodal-vision-language-model-by-hugging-face-5c437ccb0761)
- [looking for lightweight open source llms with vision capability (<2b ...](https://www.reddit.com/r/LocalLLaMA/comments/1mywvv3/looking_for_lightweight_open_source_llms_with)
- [Best Local Vision-Language Models for Offline AI - Roboflow Blog](https://blog.roboflow.com/local-vision-language-models)
- [huggingface/nanoVLM: The simplest, fastest repository for ... - GitHub](https://github.com/huggingface/nanoVLM)
- [Locally Run Huggingface LLMs like Llama on Your Laptop or ...](https://www.youtube.com/watch?v=-Fcb7OT-uC8)
- [LFM2-VL: Efficient Vision-Language Models - Liquid AI](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models)
- [How to Deploy Your LLM to Hugging Face Spaces - KDnuggets](https://www.kdnuggets.com/how-to-deploy-your-llm-to-hugging-face-spaces)
- [Image-to-Text Models – Hugging Face](https://huggingface.co/models?pipeline_tag=image-to-text&amp;sort=trending)
- [Models - Hugging Face](https://huggingface.co/models?search=ocr&sort=trending)
- [Best small vision LLM for OCR? : r/LocalLLaMA - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1f71k60/best_small_vision_llm_for_ocr)
- [Supercharge your OCR Pipelines with Open Models - Hugging Face](https://huggingface.co/blog/ocr-open-models)
- [The best open source OCR models - getomni.ai](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr)
- [10 Awesome OCR Models for 2025 - KDnuggets](https://www.kdnuggets.com/10-awesome-ocr-models-for-2025)
- [The Best Open-source OCR model | AI & ML Monthly - YouTube](https://www.youtube.com/watch?v=AhmW1t9Yw0o)
- [OCR Models - Optical Character Recognition & Text Extraction](https://huggingface.co/collections/mindchain/ocr-models-optical-character-recognition-and-text-extraction)
- [c1utchforward/OCR_For_Captchas: Finetuning based OCR solution](https://github.com/mbappeenjoyer/OCR_For_Captchas)
- [8 Top Open-Source OCR Models Compared: A Complete Guide](https://modal.com/blog/8-top-open-source-ocr-models-compared)
- [anuashok/ocr-captcha-v2 - Hugging Face](https://huggingface.co/anuashok/ocr-captcha-v2)
- [Image-to-Text Models – Hugging Face](https://huggingface.co/models?pipeline_tag=image-to-text&amp;sort=downloads)