# **MiniMax-H3-Turbo-LoRA-Fast**

MiniMax-H3-Turbo-LoRA-Fast is the denoising half of a modular, split-architecture deployment designed for high-resolution, long-form video generation (up to 20 seconds) with native audio support. Because full MiniMax-H3 model weights total nearly 200 GiB in bfloat16, this deployment splits execution across environments: the 62.14 GiB Qwen3-VL text encoder runs inside an external conditioning Space (`multimodalart/qwen3vl-conditioner`), while this engine loads the core transformer backbones and visual/audio VAEs to execute sampling and decoding loops.

The engine integrates dynamic in-memory LoRA weight folding (supporting *larry*, *lightx*, *lightx8*, *realism*, *joyfox*, and *H3-Facial-Realism-CloseUp*), aspect-ratio auto-fitting, keyframe conditioning (first-frame, last-frame, or both), and multi-pass clip chaining via FFmpeg concatenation.

### **Key Features**

* **Split-Architecture Execution:** Decouples heavy visual-language text conditioning (`Qwen3-VL`) from temporal denoising, keeping resident storage and VRAM well within GPU allocation bounds.
* **In-Place LoRA Weight Folding:** Merges dynamic LoRA weights directly into the base bfloat16 parameters at runtime without wrapper overhead, supporting fast switching between turbo and realism style adapters.
* **First & Last Frame Conditioning:** Full keyframe flexibility supporting text-to-video, image-to-video (first frame), generation toward a target state (last frame only), or bidirectional interpolation (first + last frame).
* **Dual Duration Modes:**
* `single`: Continuous single-pass generation up to 20 seconds.
* `chain`: Automatic multi-clip segmentation and progressive keyframe handoff using chunk windows (up to 14s each) merged with stream-copied FFmpeg concatenation.
* **Audio-Aware Video Decoding:** Full audio synthesis pipeline support with an isolated, host-managed mute execution path (`MiniMaxH3MuteGeneratorBlocks`) when audio generation is toggled off.

### **Repository Structure**

```text
├── app.py
├── h3_lora.py
├── h3_split_blocks.py
├── packages.txt
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To configure the environment locally, ensure you are running a modern Linux environment with a CUDA-enabled GPU and appropriate system packages.

* **Python Version:** Python **3.10+** (Python **3.11** or **3.12** recommended).
* **PyTorch Version:** `torch==2.11.0` with CUDA 13.0 support (`cu130`).
* **System Utilities:** `ffmpeg` (required for frame muxing and multi-clip concatenation).

#### **Standard Installation**

**1. Install System Dependencies**
Install FFmpeg using your system package manager:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**2. Update Package Manager**
Upgrade `pip` to satisfy installation dependencies:

```bash
pip install "pip>=26.1.2"
```

**3. Install Core Dependencies**
Install the primary deep learning stack, specific diffusers PR commit wheels, and auxiliary libraries:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
diffusers @ git+https://github.com/huggingface/diffusers.git@665f578278365ea4a3318cb8c9b66ce6c01204b9
torch==2.11.0
torchvision==0.26.0
transformers==5.8.0
accelerate==1.14.0
huggingface-hub==1.24.0
gradio
spaces==0.51.1
av
pillow
numpy
requests
safetensors>=0.8.0
```

### **Usage**

Launch the Gradio application:

```bash
python app.py
```

Once initialized, navigate to the local URL (typically `http://127.0.0.1:7860/`).

1. **Prompt Entry:** Input your text prompt describing motion, scene dynamics, lighting, and audio details.
2. **Keyframe Upload (Optional):** Add a start image in **First Frame** and/or a target ending image in **Last Frame**. Uploaded images automatically match the nearest optimal canvas aspect ratio.
3. **Select Canvas & Duration:** Choose your target resolution/aspect ratio preset and set the length slider (2s to 20s).
4. **Choose Execution Mode:** Select `single` for continuous single-pass generation or `chain` for chunked segment generation with progressive visual handoffs.
5. **Advanced Options:** Choose your active Turbo/Style LoRA (e.g., `larry`, `lightx`, `realism`, `H3-Facial-Realism-CloseUp`), set custom generation seeds, and adjust the GPU duration allocation.
6. **Generate:** Click **Generate video** to produce the final synchronized video file.

### **Links and Source**

* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/MiniMax-H3-Turbo-LoRA-Fast.git](https://github.com/PRITHIVSAKTHIUR/MiniMax-H3-Turbo-LoRA-Fast.git)
