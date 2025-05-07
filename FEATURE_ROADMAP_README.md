# HarmonyDagger Feature Roadmap Documentation

This repository contains the complete feature roadmap documentation for extending the HarmonyDagger project. The documentation outlines the planned implementation of features inspired by the HarmonyCloak research paper while maintaining compatibility with consumer-grade hardware.

## Documentation Overview

This documentation package includes five complementary documents:

| Document | Purpose | Target Audience |
|----------|---------|----------------|
| [ROADMAP.md](ROADMAP.md) | Detailed technical specifications for each feature branch | Developers, Technical Team |
| [ROADMAP_VISUALIZATION.md](ROADMAP_VISUALIZATION.md) | Visual representation of branch dependencies | Project Managers, Developers |
| [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) | High-level overview of strategic goals | Stakeholders, Management |
| [GIT_PROJECT_PLAN.md](GIT_PROJECT_PLAN.md) | Practical Git workflow and implementation guidance | Development Team |
| [RESEARCH_CONNECTIONS.md](RESEARCH_CONNECTIONS.md) | Academic context and research principles | Research-oriented Team Members |

## Core Feature Branches

All planned feature branches focus on extending HarmonyDagger's capabilities while maintaining the core principle of accessibility to average musicians on consumer-grade hardware:

1. **`branch:windowed-frequency-alignment`** - Dynamic noise alignment with short time windows
2. **`branch:multi-track-support`** - Improved multi-channel audio processing
3. **`branch:adaptive-temporal-masking`** - Implement pre/post-masking windows
4. **`branch:mp3-compression-resilience-test`** - Ensure protection survives compression
5. **`branch:perceptual-quality-check`** - Audio quality metrics
6. **`branch:genre-aware-preset-generator`** - Genre-optimized protection settings
7. **`branch:batch-folder-cli`** - Enhanced batch processing with detailed reports
8. **`branch:spectrogram-visualizer`** - Improved visualization tools
9. **`branch:audacity-plugin-export`** - Integration with popular audio software

## Implementation Philosophy

The roadmap adheres to these key principles:

1. **Consumer Hardware Compatibility** - No GPU-intensive operations
2. **Preserving Audio Quality** - Minimal perceptual impact for human listeners
3. **Practical Usability** - Focus on real-world music protection workflows
4. **Research-Backed Methods** - Based on established psychoacoustic principles
5. **Incremental Development** - Building features in a logical, dependency-aware sequence

## Getting Started

To begin working with this feature roadmap:

1. Start by reading the [Executive Summary](EXECUTIVE_SUMMARY.md) for a high-level overview
2. Review the [Roadmap Visualization](ROADMAP_VISUALIZATION.md) to understand feature dependencies
3. Examine the detailed [Roadmap](ROADMAP.md) for technical specifications
4. Consult the [Git Project Plan](GIT_PROJECT_PLAN.md) for implementation workflow
5. Reference the [Research Connections](RESEARCH_CONNECTIONS.md) for academic context

## Contribution Guidelines

When implementing features from this roadmap:

1. Follow the branch naming conventions exactly as specified
2. Adhere to the dependency order shown in the visualization
3. Ensure each implementation maintains compatibility with consumer hardware
4. Include appropriate tests and documentation
5. Reference the original research where applicable

## License

This feature roadmap documentation is released under the same license as the main HarmonyDagger project (MIT License).

## Acknowledgments

The feature roadmap is based on the HarmonyCloak research paper by Syed Irfan Ali Meerza, Lichao Sun, and Jian Liu, which introduced the core concepts for making music unlearnable for generative AI while preserving human listening experience.
