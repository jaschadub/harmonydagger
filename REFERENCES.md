# References

Research underpinning HarmonyDagger's design. Grouped by topic; works that directly inspired specific modules are flagged with the relevant module name.

## Direct Inspirations

These works most directly shaped HarmonyDagger's approach of using imperceptible, psychoacoustically-masked perturbations to make audio unlearnable for generative AI.

- **HarmonyCloak: Making Music Unlearnable for Generative AI**
  Syed Irfan Ali Meerza, Lichao Sun, Jian Liu.
  *Proceedings of the Network and Distributed System Security Symposium (NDSS)*, 2025.
  Introduces "unlearnable" perturbations specifically targeting music generation models (e.g., MusicGen, Jukebox). Demonstrates that small, perceptually-bounded noise added to training audio prevents downstream generative models from reproducing the protected content while remaining inaudible. HarmonyDagger generalizes this idea to voice/podcast audio and combines it with ensemble strategies across spectral, mel, and embedding domains.

- **AntiFake: Using Adversarial Audio to Prevent Unauthorized Speech Synthesis**
  Zhiyuan Yu, Shixuan Zhai, Ning Zhang.
  *ACM Conference on Computer and Communications Security (CCS)*, 2023.
  Adversarially perturbs speech to prevent voice-cloning systems (encoders, vocoders) from extracting usable speaker embeddings. Motivates `vocal_mode.py` (300 Hz–3 kHz emphasis, formant targeting) and the embedding-disruption branch of `ensemble.py`.

- **VSMask: Defending Against Voice Synthesis Attack via Real-Time Predictive Perturbation**
  Yuanda Wang et al.
  *ACM Conference on Security and Privacy in Wireless and Mobile Networks (WiSec)*, 2023.
  Real-time variant of the voice-cloning defense problem. Relevant to future streaming/real-time directions.

## Adversarial Audio Foundations

- **Audio Adversarial Examples: Targeted Attacks on Speech-to-Text**
  Nicholas Carlini, David Wagner.
  *IEEE Symposium on Security and Privacy Workshops (SPW / DLS)*, 2018.
  Foundational demonstration that small perturbations can steer ASR outputs. Establishes the L∞-bounded perturbation framework that subsequent perceptual-bound work refines.

- **Imperceptible, Robust, and Targeted Adversarial Examples for Automatic Speech Recognition**
  Yao Qin, Nicholas Carlini, Garrison Cottrell, Ian Goodfellow, Colin Raffel.
  *International Conference on Machine Learning (ICML)*, 2019.
  Introduces psychoacoustic masking constraints into adversarial audio optimization — perturbations are bounded below the frequency-dependent masking threshold. Directly motivates `psychoacoustics.py`.

- **Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding**
  Lea Schönherr, Katharina Kohls, Steffen Zeiler, Thorsten Holz, Dorothea Kolossa.
  *Network and Distributed System Security Symposium (NDSS)*, 2019.
  Independent contemporaneous work hiding adversarial audio inside masking thresholds derived from the MPEG-1 psychoacoustic model. Core reference for the masking-threshold approach used throughout HarmonyDagger.

## Unlearnable Examples (Data Poisoning for Defense)

- **Unlearnable Examples: Making Personal Data Unexploitable**
  Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang.
  *International Conference on Learning Representations (ICLR)*, 2021.
  Image-domain origin of "unlearnable" perturbations: error-minimizing noise prevents models from learning useful features from protected data. HarmonyCloak adapts this idea to audio; HarmonyDagger inherits the threat model.

- **Availability Attacks Create Shortcuts**
  Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, Tie-Yan Liu.
  *ACM Conference on Knowledge Discovery and Data Mining (KDD)*, 2022.
  Theoretical framing of why unlearnable perturbations work — they introduce "shortcut features" the model latches onto instead of true signal.

## Psychoacoustic Modeling

These works ground the hearing-threshold, Bark-scale, and masking computations in `psychoacoustics.py`, `temporal_masking.py`, and `common.py`.

- **ISO 226:2003 — Normal Equal-Loudness-Level Contours.**
  International Organization for Standardization.
  Source of the absolute hearing threshold curve approximated in `hearing_threshold()`.

- **Psychoacoustics: Facts and Models** (3rd ed.)
  Eberhard Zwicker, Hugo Fastl. Springer, 2007.
  Canonical reference for the Bark scale, critical bandwidths, and simultaneous/temporal masking models used throughout this codebase.

- **An Introduction to the Psychology of Hearing** (6th ed.)
  Brian C. J. Moore. Brill, 2012.
  Standard treatment of forward (post-) masking dynamics — ~200 ms decay window with rapid initial fall — implemented in `temporal_masking.py`.

- **Perceptual Coding of Digital Audio**
  Ted Painter, Andreas Spanias.
  *Proceedings of the IEEE*, vol. 88, no. 4, 2000.
  Comprehensive survey of psychoacoustic models used in MP3/AAC. Practical reference for translating ISO standards into runnable code.

- **ISO/IEC 11172-3:1993 — MPEG-1 Audio (Psychoacoustic Models 1 and 2).**
  Defines the masking-threshold computation procedure that perceptual audio codecs (and several adversarial-audio works) use as their hiding budget.

- **Subdivision of the Audible Frequency Range into Critical Bands (Frequenzgruppen)**
  Eberhard Zwicker.
  *Journal of the Acoustical Society of America*, vol. 33, no. 2, 1961.
  Origin of the Bark scale.

## Phase Perception

Background for `phase.py` (subtle phase shifts that disrupt ML feature extraction while remaining inaudible).

- **The Importance of Phase in Signals**
  Alan V. Oppenheim, Jae S. Lim.
  *Proceedings of the IEEE*, vol. 69, no. 5, 1981.
  Classic argument for phase carrying significant perceptual information in some regimes — motivates why phase perturbations affect ML features.

- **The Importance of Phase in Speech Enhancement**
  Kuldip Paliwal, Kamil Wójcicki, Benjamin Shannon.
  *Speech Communication*, vol. 53, no. 4, 2011.
  Quantifies when phase matters for speech tasks. Useful for choosing `PHASE_PERTURBATION_MAX_RADIANS` bounds.

## AI Audio Generation Models (Targets of Protection)

HarmonyDagger aims to disrupt training/fine-tuning of these model families. The ensemble strategies in `ensemble.py` are chosen to span the input representations these systems use.

- **WaveNet: A Generative Model for Raw Audio**
  Aäron van den Oord et al. DeepMind, 2016.
  Raw-waveform autoregressive generation.

- **Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions**
  Jonathan Shen et al. *ICASSP*, 2018.
  Mel-spectrogram TTS pipeline — informs the `mel_band` ensemble branch.

- **HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis**
  Jungil Kong, Jaehyeon Kim, Jaekyoung Bae. *NeurIPS*, 2020.
  STFT-domain GAN vocoder — informs the `spectral` ensemble branch.

- **Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)**
  Jaehyeon Kim, Jungil Kong, Juhee Son. *ICML*, 2021.
  End-to-end TTS with stochastic duration prediction.

- **MusicGen: Simple and Controllable Music Generation**
  Jade Copet et al. Meta AI. *NeurIPS*, 2023.
  Transformer-based music generation. One of the primary HarmonyCloak targets.

- **AudioLDM 2: Learning Holistic Audio Generation with Self-Supervised Pretraining**
  Haohe Liu et al. *ICML*, 2024.
  Diffusion-based audio generation operating on latent mel representations.

- **Jukebox: A Generative Model for Music**
  Prafulla Dhariwal et al. OpenAI, 2020.
  VQ-VAE + transformer music generation.

## Voice Cloning Systems (Vocal-Mode Threat Model)

- **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS)**
  Ye Jia et al. *NeurIPS*, 2018.
  Zero-shot voice cloning from a few seconds of reference audio — the canonical threat that `vocal_mode.py` addresses.

- **YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion**
  Edresson Casanova et al. *ICML*, 2022.

- **VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers**
  Chengyi Wang et al. Microsoft, 2023.

## Robustness and Survivability

Background for `robustness.py` (testing whether perturbations survive MP3 compression, low-pass filtering, resampling).

- **Expectation Over Transformation for Robust Adversarial Examples**
  Anish Athalye, Logan Engstrom, Andrew Ilyas, Kevin Kwok. *ICML*, 2018.
  Foundational technique for crafting perturbations that survive a known distribution of input transforms. Worth adopting if/when HarmonyDagger moves from "test survival" to "optimize for survival."

- **MP3 Compression as a Defense Against Adversarial Examples**
  Various — see e.g. Das et al. *arXiv:1705.02900*, 2017.
  Documents how lossy compression strips many adversarial perturbations. Motivates the MP3 round-trip check in `robustness.py`.

## Evaluation Metrics

- **Perceptual Evaluation of Speech Quality (PESQ).** ITU-T Recommendation P.862.
- **Perceptual Evaluation of Audio Quality (PEAQ).** ITU-R Recommendation BS.1387.
  Both are candidates for tighter perceptual-quality reporting in `benchmark.py` and `verify.py`; current implementation uses SNR + MFCC distance as lightweight proxies.

## Related Defensive Tools

For context on the broader ecosystem of "protect media from generative AI" tools:

- **Glaze** — Shawn Shan et al. *USENIX Security*, 2023. Style-mimicry defense for visual art.
- **Nightshade** — Shawn Shan et al. *IEEE S&P*, 2024. Poisoning attack on text-to-image training.
- **PhotoGuard** — Hadi Salman et al. *ICML*, 2023. Disrupts image editing by diffusion models.

HarmonyDagger occupies the analogous niche for audio, drawing methodological inspiration from the audio-specific (HarmonyCloak, AntiFake) line of work rather than from these image-domain tools.
