# HarmonyDagger Git Project Plan

This document provides practical Git workflow guidelines for implementing the feature branches outlined in the roadmap.

## Branch Naming and Creation

All feature branches should follow the exact naming pattern specified in the roadmap:

```bash
# Example branch creation
git checkout main
git pull
git checkout -b windowed-frequency-alignment
```

## Git Workflow Guidelines

### 1. Branch Management

- **Keep branches focused**: Each branch should implement only the specific feature it's named for
- **Regular commits**: Make incremental commits with clear messages
- **Rebasing**: Regularly rebase feature branches on main to incorporate updates

### 2. Commit Message Format

Use the following format for commit messages:

```
[Feature] Brief description (50 chars max)

More detailed explanation of what changed and why.
Include relevant details about implementation decisions.

Refs: #issue-number (if applicable)
```

Example:
```
[Windowed-Freq] Implement peak detection algorithm

Add multi-peak detection with moving average smoothing to 
improve frequency alignment in short time frames. The window
size is now configurable from 5-20ms with 10ms as default.

Refs: #14
```

### 3. Pull Request Template

When creating a pull request, include:

1. **Feature summary**: Brief description of implemented feature
2. **Technical details**: Key implementation decisions
3. **Testing**: How the feature was tested
4. **Screenshots**: If applicable (especially for visualization features)
5. **Dependencies**: Which other branches this PR depends on

### 4. Branch Dependencies and Merge Order

Follow the dependency chart in ROADMAP_VISUALIZATION.md for proper branch sequencing.

General rules:
- Merge core algorithm improvements first
- Quality metrics branches next
- File handling and visualization after that
- User experience features after core functionality is stable
- Distribution channels last

### 5. Development Phases

For each branch, follow these phases:

#### Phase 1: Research & Design
- Review relevant academic papers and existing code
- Create design document with algorithms to implement
- Get feedback on design before implementation

#### Phase 2: Implementation
- Implement core functionality
- Add tests to verify behavior
- Document new features

#### Phase 3: Testing & Validation
- Test across different audio types
- Measure performance impact
- Verify audio quality is maintained
- Check compatibility with other features

#### Phase 4: Integration
- Prepare pull request with documentation
- Address review feedback
- Merge to main

## Branch-Specific Development Notes

### `branch:windowed-frequency-alignment`

**Key implementation considerations:**
- Need unit tests for windowing algorithm
- Verify with spectrograms that alignment is correct
- Benchmark performance impact of different window sizes

### `branch:adaptive-temporal-masking`

**Key implementation considerations:**
- Need good test cases with transient-heavy audio
- Validate with listening tests that masking is effective
- Ensure the algorithm doesn't introduce audible artifacts

### `branch:perceptual-quality-check`

**Key implementation considerations:**
- Need validation against established perceptual metrics
- Generate test suite with diverse audio samples
- Build visualization tools for quality scores

### Feature Gating Strategy

Consider using feature flags to:
- Include experimental features in the codebase before they're ready
- Allow users to opt-in to testing new features
- A/B test different algorithm variants

```python
# Example feature flag implementation
FEATURE_FLAGS = {
    'use_windowed_frequency': True,
    'use_temporal_masking': True,
    'enable_compression_test': False,  # Not ready for production yet
}

# Usage in code
if FEATURE_FLAGS['use_temporal_masking']:
    apply_temporal_masking(audio)
```

## Resources and Time Allocation

To successfully implement all branches, allocate time as follows:

1. **Core Algorithm Improvements**: ~40% of development time
   - Windowed frequency alignment: 20%
   - Adaptive temporal masking: 20%
   
2. **Quality Metrics**: ~20% of development time
   - Perceptual quality check: 10%
   - MP3 compression resilience test: 10%
   
3. **File Handling & Visualization**: ~20% of development time
   - Multi-track support: 10%
   - Spectrogram visualizer: 10%
   
4. **User Experience**: ~15% of development time
   - Genre-aware preset generator: 5%
   - Batch folder CLI: 10%
   
5. **Distribution**: ~5% of development time
   - Audacity plugin export

This allocation ensures sufficient focus on the core algorithm improvements while still delivering a complete feature set.
