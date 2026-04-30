# Primus: Transformer-centric 3D medical image segmentation

Primus is a line of Transformer-centric architectures for 3D medical image segmentation. It was developed to answer a simple question: *can Transformers become genuinely competitive for volumetric medical segmentation, without hiding most of the work inside a CNN backbone?*

Most existing “Transformer-based” segmentation networks in medical imaging are CNN–Transformer hybrids. In our analysis, many of these architectures allocate substantial capacity outside their Transformer blocks, and several show little or no performance degradation when the Transformer blocks are replaced by identity mappings. This suggests that, in many popular hybrids, the CNN components can solve the task largely on their own.

Primus addresses this by moving the architecture back toward the Vision Transformer paradigm. The model keeps almost all parameters and FLOPs inside the Transformer backbone, uses high-resolution 3D tokens, and relies on a lightweight decoder. PrimusV2 extends this design with a stronger iterative tokenizer, improving the quality of visual tokens while preserving a measurable dependence on the Transformer.

## Key ideas

- **Transformer-centric design:** Primus minimizes non-Transformer capacity, ensuring that representation learning is driven primarily by attention.
- **High-resolution 3D tokens:** Instead of the common `16 × 16 × 16` tokenization, Primus uses `8 × 8 × 8` tokens to preserve local anatomical detail.
- **3D rotary positional embeddings:** Axial 3D RoPE improves spatial awareness under patch-based 3D training.
- **Modern Transformer blocks:** Primus uses design elements such as SwiGLU MLPs, LayerScale, post-attention normalization, DropPath, AdamW, and strong weight decay.
- **nnU-Net integration:** Primus is implemented inside the nnU-Net framework and can use its preprocessing, planning, augmentation, and training infrastructure.
- **PrimusV2 tokenizer:** PrimusV2 adds a minimal residual iterative tokenizer, which substantially improves performance on small and fine-grained structures while keeping the Transformer functionally important.

## Architecture overview

![Primus architecture overview](documentation/assets/primusv2.png?raw=true)


Primus tokenizes a 3D input volume into high-resolution visual tokens, processes them with a Transformer backbone using 3D axial RoPE and modern Transformer blocks, and decodes the resulting token sequence back into a dense segmentation map. PrimusV2 follows the same overall design, but replaces the single strided projection with an iterative residual tokenizer to produce stronger visual tokens.

## Why Primus?

Transformers are attractive for 3D medical imaging because they naturally operate on token sequences. This makes them well-suited for scalable self-supervised learning, masked image modeling, and future multimodal systems that combine images, text, reports, measurements, and other clinical data. However, these benefits require a Transformer that actually learns strong 3D visual tokens.

Primus provides such a backbone. It closes the gap between Transformer-based segmentation models and strong CNN baselines. PrimusV2 goes further and reaches parity with state-of-the-art convolutional architectures such as ResEnc-L and MedNeXt across a diverse set of public 3D medical segmentation datasets.

## Architecture family

Primus is available in multiple scales:

| Model | Layers | Heads | Embedding dim. | Main use case |
|---|---:|---:|---:|---|
| Primus-S | 12 | 6 | 396 | Lightweight experiments and smaller datasets |
| Primus-B | 12 | 12 | 792 | Balanced baseline |
| Primus-M | 16 | 12 | 864 | Recommended default |
| Primus-L | 24 | 16 | 1056 | Large-scale experiments |

PrimusV2 follows the same scaling philosophy but replaces the single strided patch projection with an iterative residual tokenizer.

## Test set results

The following table summarizes the test-set Dice similarity coefficient (DSC) results for Primus, PrimusV2, and selected baselines. Values should be reported as mean DSC over the corresponding cross-validation folds.

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Params. (M)</th>
      <th>ACDC</th>
      <th>AMOS22</th>
      <th>KiTS23</th>
      <th>LiTS</th>
      <th>SST3</th>
      <th>MAMA</th>
      <th>SBM</th>
      <th>Atlas22</th>
      <th>WORD</th>
      <th style="border-left: 2px solid #999;">Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>nnU-Net</td>
      <td>31.2</td>
      <td>91.34</td>
      <td>88.61</td>
      <td>85.99</td>
      <td>79.29</td>
      <td>90.27</td>
      <td>78.32</td>
      <td>66.52</td>
      <td>63.11</td>
      <td>83.11</td>
      <td style="border-left: 2px solid #999;">80.73</td>
    </tr>
    <tr>
      <td>ResEnc-L</td>
      <td>102.4</td>
      <td>92.54</td>
      <td>89.39</td>
      <td>88.06</td>
      <td>81.20</td>
      <td>90.28</td>
      <td>79.00</td>
      <td>64.00</td>
      <td>63.12</td>
      <td>85.79</td>
      <td style="border-left: 2px solid #999;">81.48</td>
    </tr>
    <tr>
      <td>MedNeXt-L</td>
      <td>61.8</td>
      <td>92.55</td>
      <td>89.58</td>
      <td>88.20</td>
      <td>81.57</td>
      <td>89.93</td>
      <td>79.42</td>
      <td>65.85</td>
      <td>63.03</td>
      <td>85.37</td>
      <td style="border-left: 2px solid #999;">81.72</td>
    </tr>
    <tr>
      <td>CoTr</td>
      <td>41.9</td>
      <td>90.50</td>
      <td>87.93</td>
      <td>84.63</td>
      <td>78.44</td>
      <td>89.60</td>
      <td>76.95</td>
      <td>59.96</td>
      <td>62.14</td>
      <td>83.11</td>
      <td style="border-left: 2px solid #999;">79.25</td>
    </tr>
    <tr>
      <td>Primus-M</td>
      <td>146.6</td>
      <td>92.26</td>
      <td>88.18</td>
      <td>86.38</td>
      <td>79.52</td>
      <td>88.31</td>
      <td>76.39</td>
      <td>57.63</td>
      <td>60.10</td>
      <td>82.98</td>
      <td style="border-left: 2px solid #999;">79.08</td>
    </tr>
    <tr>
      <td>PrimusV2-M</td>
      <td>147.2</td>
      <td>92.27</td>
      <td>89.35</td>
      <td>88.09</td>
      <td>81.73</td>
      <td>88.26</td>
      <td>79.40</td>
      <td>66.36</td>
      <td>63.23</td>
      <td>84.15</td>
      <td style="border-left: 2px solid #999;">81.43</td>
    </tr>
  </tbody>
</table>

## Primus trainers
> Currently only the default Primus trainers are implemented as nnU-Net trainers.
> PrimusV2 (and PrimusV3) trainers will be added in the near future.

The following Primus trainers are available:
```text
nnUNet_Primus_S_Trainer
nnUNet_Primus_B_Trainer
nnUNet_Primus_M_Trainer
nnUNet_Primus_L_Trainer
```

