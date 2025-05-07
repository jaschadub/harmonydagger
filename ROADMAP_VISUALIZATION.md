# HarmonyDagger Feature Branch Visualization

This diagram illustrates the planned feature branches for HarmonyDagger, showing implementation dependencies and relationships.

## Branch Implementation Sequence

```mermaid
flowchart TD
    main["main branch (current)"]
    
    %% Core algorithm improvements
    wfa["branch:windowed-frequency-alignment
    Dynamic frequency noise alignment"]
    atm["branch:adaptive-temporal-masking
    Pre/post masking model"]
    
    %% Quality & analysis features
    pqc["branch:perceptual-quality-check
    Audio quality metrics"]
    mct["branch:mp3-compression-resilience-test
    Compression survival testing"]
    
    %% File handling & visualization
    mts["branch:multi-track-support
    Enhanced multi-channel processing"]
    spv["branch:spectrogram-visualizer
    Enhanced visualization tools"]
    
    %% User experience & distribution
    gap["branch:genre-aware-preset-generator
    Genre-specific parameter presets"]
    bfc["branch:batch-folder-cli
    Enhanced batch processing"]
    ape["branch:audacity-plugin-export
    DAW integration"]

    %% Define dependencies/sequence
    main --> wfa
    main --> atm
    
    wfa --> pqc
    atm --> pqc
    
    pqc --> mct
    
    wfa --> mts
    mct --> mts
    
    pqc --> spv
    
    pqc --> gap
    mts --> gap
    
    gap --> bfc
    
    bfc --> ape
    spv --> ape
    
    %% Styling
    classDef core fill:#f9d5e5,stroke:#333,stroke-width:2px;
    classDef quality fill:#eeeeee,stroke:#333,stroke-width:2px;
    classDef handling fill:#e3eaa7,stroke:#333,stroke-width:2px;
    classDef ux fill:#b5ead7,stroke:#333,stroke-width:2px;
    classDef main fill:#c7ceea,stroke:#333,stroke-width:2px;
    
    class main main;
    class wfa,atm core;
    class pqc,mct quality;
    class mts,spv handling;
    class gap,bfc,ape ux;
```

## Feature Categories

- **Core Algorithm Improvements** (pink): Fundamental enhancements to noise injection methods
- **Quality & Analysis** (gray): Features that measure and ensure effectiveness
- **File Handling & Visualization** (light green): Support for different audio formats and better visualization
- **User Experience & Distribution** (mint): Features that improve usability and adoption

## Implementation Timeline Considerations

1. Core algorithm improvements should be implemented first as they affect all other features
2. Quality metrics are needed next to validate the effectiveness of protection
3. File handling and visualization features enhance usability
4. User experience features like presets and batch processing streamline workflows
5. The Audacity plugin should be developed last to incorporate all other features

## Dependencies Explained

- **Perceptual quality check** depends on core algorithm improvements to measure their effectiveness
- **MP3 compression resilience test** builds on quality metrics
- **Multi-track support** benefits from both frequency alignment and compression testing
- **Genre-aware presets** require quality metrics and multi-track support
- **Batch folder CLI** builds on genre presets for more effective batch processing
- **Audacity plugin** incorporates batch processing capabilities and visualization tools
