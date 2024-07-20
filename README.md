# awesome-3DGS

3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities

#### 3D Gaussian Splatting for Real-Time Radiance Field Rendering

**Authors**: George Drettakis, Thomas Leimkühler, Georgios Kopanas, Bernhard Kerbl

<details>
<summary><b>Abstract</b></summary>
Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.04079v1.pdf)

# Optimization of 3D Gaussian Splatting
## Efficiency
### Storage Efficiency
#### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

**Authors**: Bo Dai, Dahua Lin, LiMin Wang, Yuanbo Xiangli, Linning Xu, Mulin Yu, Tao Lu

<details>
<summary><b>Abstract</b></summary>
Neural rendering methods have significantly advanced photo-realistic 3D scene rendering in various academic and industrial applications. The recent 3D Gaussian Splatting method has achieved the state-of-the-art rendering quality and speed combining the benefits of both primitive-based representations and volumetric representations. However, it often leads to heavily redundant Gaussians that try to fit every training view, neglecting the underlying scene geometry. Consequently, the resulting model becomes less robust to significant view changes, texture-less area and lighting effects. We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum. Anchor growing and pruning strategies are developed based on the importance of neural Gaussians to reliably improve the scene coverage. We show that our method effectively reduces redundant Gaussians while delivering high-quality rendering. We also demonstrates an enhanced capability to accommodate scenes with varying levels-of-detail and view-dependent observations, without sacrificing the rendering speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00109v1.pdf)

#### LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

**Authors**: Zhangyang Wang, Dejia Xu, Zehao Zhu, Kairun Wen, Kevin Wang, Zhiwen Fan

<details>
<summary><b>Abstract</b></summary>
Recent advancements in real-time neural rendering using point-based techniques have paved the way for the widespread adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting come with a substantial storage overhead caused by growing the SfM points to millions, often demanding gigabyte-level disk space for a single unbounded scene, posing significant scalability challenges and hindering the splatting efficiency. To address this challenge, we introduce LightGaussian, a novel method designed to transform 3D Gaussians into a more efficient and compact format. Drawing inspiration from the concept of Network Pruning, LightGaussian identifies Gaussians that are insignificant in contributing to the scene reconstruction and adopts a pruning and recovery process, effectively reducing redundancy in Gaussian counts while preserving visual effects. Additionally, LightGaussian employs distillation and pseudo-view augmentation to distill spherical harmonics to a lower degree, allowing knowledge transfer to more compact representations while maintaining reflectance. Furthermore, we propose a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in lower bitwidth representations with minimal accuracy losses. In summary, LightGaussian achieves an averaged compression rate over 15x while boosting the FPS from 139 to 215, enabling an efficient representation of complex scenes on Mip-NeRF 360, Tank and Temple datasets. Project website: https://lightgaussian.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17245v5.pdf)

#### Compact 3D Scene Representation via Self-Organizing Gaussian Grids

**Authors**: Peter Eisert, Anna Hilsmann, Florian Barthel, Wieland Morgenstern

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting has recently emerged as a highly promising technique for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it utilizes efficient rasterization allowing for very fast rendering at high-quality. However, the storage size is significantly higher, which hinders practical deployment, e.g. on resource constrained devices. In this paper, we introduce a compact scene representation organizing the parameters of 3D Gaussian Splatting (3DGS) into a 2D grid with local homogeneity, ensuring a drastic reduction in storage requirements without compromising visual quality during rendering. Central to our idea is the explicit exploitation of perceptual redundancies present in natural scenes. In essence, the inherent nature of a scene allows for numerous permutations of Gaussian parameters to equivalently represent it. To this end, we propose a novel highly parallel algorithm that regularly arranges the high-dimensional Gaussian parameters into a 2D grid while preserving their neighborhood structure. During training, we further enforce local smoothness between the sorted parameters in the grid. The uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless integration with established renderers. Our method achieves a reduction factor of 17x to 42x in size for complex scenes with no increase in training time, marking a substantial leap forward in the domain of 3D scene distribution and consumption. Additional information can be found on our project page: https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13299v2.pdf)

#### Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

**Authors**: Alexander Keller, Christoph Schied, Alex Evans, Thomas Müller

<details>
<summary><b>Abstract</b></summary>
Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent. The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs. We leverage this parallelism by implementing the whole system using fully-fused CUDA kernels with a focus on minimizing wasted bandwidth and compute operations. We achieve a combined speedup of several orders of magnitude, enabling training of high-quality neural graphics primitives in a matter of seconds, and rendering in tens of milliseconds at a resolution of ${1920\!\times\!1080}$.
</details>

[📄 Paper](https://arxiv.org/pdf/2201.05989v2.pdf)

#### Compact 3D Gaussian Representation for Radiance Field

**Authors**: Eunbyung Park, Jong Hwan Ko, Xiangyu Sun, Daniel Rho, Joo Chan Lee

<details>
<summary><b>Abstract</b></summary>
Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussisan-based representation and adopts the rasterization pipeline to render the images rather than volumetric rendering, achieving very fast rendering speed and promising image quality. However, a significant drawback arises as 3DGS entails a substantial number of 3D Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric attributes of Gaussian by vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25$\times$ reduced storage and enhanced rendering speed, while maintaining the quality of the scene representation, compared to 3DGS. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13681v2.pdf)

#### GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering

**Authors**: Guocheng Qian, Jinjie Mai, Andrea Vedaldi, Bernard Ghanem, Carl Vondrick, Ruoshi Liu, Luke Melas-Kyriazi, Abdullah Hamdi

<details>
<summary><b>Abstract</b></summary>
Advancements in 3D Gaussian Splatting have significantly accelerated 3D reconstruction and generation. However, it may require a large number of Gaussians, which creates a substantial memory footprint. This paper introduces GES (Generalized Exponential Splatting), a novel representation that employs Generalized Exponential Function (GEF) to model 3D scenes, requiring far fewer particles to represent a scene and thus significantly outperforming Gaussian Splatting methods in efficiency with a plug-and-play replacement ability for Gaussian-based utilities. GES is validated theoretically and empirically in both principled 1D setup and realistic 3D scenes. It is shown to represent signals with sharp edges more accurately, which are typically challenging for Gaussians due to their inherent low-pass characteristics. Our empirical analysis demonstrates that GEF outperforms Gaussians in fitting natural-occurring signals (e.g. squares, triangles, and parabolic signals), thereby reducing the need for extensive splitting operations that increase the memory footprint of Gaussian Splatting. With the aid of a frequency-modulated loss, GES achieves competitive performance in novel-view synthesis benchmarks while requiring less than half the memory storage of Gaussian Splatting and increasing the rendering speed by up to 39%. The code is available on the project website https://abdullahamdi.com/ges .
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10128v2.pdf)

#### Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

**Authors**: Rüdiger Westermann, Josef Stumpfegger, Simon Niedermayr

<details>
<summary><b>Abstract</b></summary>
Recently, high-fidelity scene reconstruction with an optimized 3D Gaussian splat representation has been introduced for novel view synthesis from sparse image sets. Making such representations suitable for applications like network streaming and rendering on low-power devices requires significantly reduced memory consumption as well as improved rendering efficiency. We propose a compressed 3D Gaussian splat representation that utilizes sensitivity-aware vector clustering with quantization-aware training to compress directional colors and Gaussian parameters. The learned codebooks have low bitrates and achieve a compression rate of up to $31\times$ on real-world scenes with only minimal degradation of visual quality. We demonstrate that the compressed splat representation can be efficiently rendered with hardware rasterization on lightweight GPUs at up to $4\times$ higher framerates than reported via an optimized GPU compute pipeline. Extensive experiments across multiple datasets demonstrate the robustness and rendering speed of the proposed approach.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.02436v2.pdf)

#### Compact3d: Compressing gaussian splat radiance field models with vector quantization
  - **Not Found on Arxiv by arxiv api, will be append manually later**

### Training Efficiency
#### DISTWAR: Fast Differentiable Rendering on Raster-based Rendering Pipelines

**Authors**: Nandita Vijaykumar, Pawan Kumar Sanjaya, Ruofan Liang, Fan Chen, Adrian Zhao, Sankeerth Durvasula

<details>
<summary><b>Abstract</b></summary>
Differentiable rendering is a technique used in an important emerging class of visual computing applications that involves representing a 3D scene as a model that is trained from 2D images using gradient descent. Recent works (e.g. 3D Gaussian Splatting) use a rasterization pipeline to enable rendering high quality photo-realistic imagery at high speeds from these learned 3D models. These methods have been demonstrated to be very promising, providing state-of-art quality for many important tasks. However, training a model to represent a scene is still a time-consuming task even when using powerful GPUs. In this work, we observe that the gradient computation phase during training is a significant bottleneck on GPUs due to the large number of atomic operations that need to be processed. These atomic operations overwhelm atomic units in the L2 partitions causing stalls. To address this challenge, we leverage the observations that during the gradient computation: (1) for most warps, all threads atomically update the same memory locations; and (2) warps generate varying amounts of atomic traffic (since some threads may be inactive). We propose DISTWAR, a software-approach to accelerate atomic operations based on two key ideas: First, we enable warp-level reduction of threads at the SM sub-cores using registers to leverage the locality in intra-warp atomic updates. Second, we distribute the atomic computation between the warp-level reduction at the SM and the L2 atomic units to increase the throughput of atomic computation. Warps with many threads performing atomic updates to the same memory locations are scheduled at the SM, and the rest using L2 atomic units. We implement DISTWAR using existing warp-level primitives. We evaluate DISTWAR on widely used raster-based differentiable rendering workloads. We demonstrate significant speedups of 2.44x on average (up to 5.7x).
</details>
[📄 Paper](https://arxiv.org/pdf/2401.05345v1.pdf)

### Rendering Efficiency
#### Identifying Unnecessary 3D Gaussians using Clustering for Fast Rendering of 3D Gaussian Splatting

**Authors**: Jongsun Park, Hyeongwon Kim, Joongho Jo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting (3D-GS) is a new rendering approach that outperforms the neural radiance field (NeRF) in terms of both speed and image quality. 3D-GS represents 3D scenes by utilizing millions of 3D Gaussians and projects these Gaussians onto the 2D image plane for rendering. However, during the rendering process, a substantial number of unnecessary 3D Gaussians exist for the current view direction, resulting in significant computation costs associated with their identification. In this paper, we propose a computational reduction technique that quickly identifies unnecessary 3D Gaussians in real-time for rendering the current view without compromising image quality. This is accomplished through the offline clustering of 3D Gaussians that are close in distance, followed by the projection of these clusters onto a 2D image plane during runtime. Additionally, we analyze the bottleneck associated with the proposed technique when executed on GPUs and propose an efficient hardware architecture that seamlessly supports the proposed scheme. For the Mip-NeRF360 dataset, the proposed technique excludes 63% of 3D Gaussians on average before the 2D image projection, which reduces the overall rendering computation by almost 38.3% without sacrificing peak-signal-to-noise-ratio (PSNR). The proposed accelerator also achieves a speedup of 10.7x compared to a GPU.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.13827v1.pdf)

#### GSCore: Efficient Radiance Field Rendering via Architectural Support for 3D Gaussian Splatting
  - **Not Found on Arxiv by arxiv api, will be append manually later**

## Photorealism
#### Microfacet models for refraction through rough surfaces
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### Mirror-3DGS: Incorporating Mirror Reflections into 3D Gaussian Splatting

**Authors**: Siwei Ma, Jian Zhang, Shuzhou Yang, Qiankun Gao, Yanmin Wu, Haijie Li, Jiarui Meng

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has marked a significant breakthrough in the realm of 3D scene reconstruction and novel view synthesis. However, 3DGS, much like its predecessor Neural Radiance Fields (NeRF), struggles to accurately model physical reflections, particularly in mirrors that are ubiquitous in real-world scenes. This oversight mistakenly perceives reflections as separate entities that physically exist, resulting in inaccurate reconstructions and inconsistent reflective properties across varied viewpoints. To address this pivotal challenge, we introduce Mirror-3DGS, an innovative rendering framework devised to master the intricacies of mirror geometries and reflections, paving the way for the generation of realistically depicted mirror reflections. By ingeniously incorporating mirror attributes into the 3DGS and leveraging the principle of plane mirror imaging, Mirror-3DGS crafts a mirrored viewpoint to observe from behind the mirror, enriching the realism of scene renderings. Extensive assessments, spanning both synthetic and real-world scenes, showcase our method's ability to render novel views with enhanced fidelity in real-time, surpassing the state-of-the-art Mirror-NeRF specifically within the challenging mirror regions. Our code will be made publicly available for reproducible research.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.01168v1.pdf)

#### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

**Authors**: Bo Dai, Dahua Lin, LiMin Wang, Yuanbo Xiangli, Linning Xu, Mulin Yu, Tao Lu

<details>
<summary><b>Abstract</b></summary>
Neural rendering methods have significantly advanced photo-realistic 3D scene rendering in various academic and industrial applications. The recent 3D Gaussian Splatting method has achieved the state-of-the-art rendering quality and speed combining the benefits of both primitive-based representations and volumetric representations. However, it often leads to heavily redundant Gaussians that try to fit every training view, neglecting the underlying scene geometry. Consequently, the resulting model becomes less robust to significant view changes, texture-less area and lighting effects. We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum. Anchor growing and pruning strategies are developed based on the importance of neural Gaussians to reliably improve the scene coverage. We show that our method effectively reduces redundant Gaussians while delivering high-quality rendering. We also demonstrates an enhanced capability to accommodate scenes with varying levels-of-detail and view-dependent observations, without sacrificing the rendering speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00109v1.pdf)

#### Mip-Splatting: Alias-free 3D Gaussian Splatting

**Authors**: Andreas Geiger, Torsten Sattler, Binbin Huang, Anpei Chen, Zehao Yu

<details>
<summary><b>Abstract</b></summary>
Recently, 3D Gaussian Splatting has demonstrated impressive novel view synthesis results, reaching high fidelity and efficiency. However, strong artifacts can be observed when changing the sampling rate, \eg, by changing focal length or camera distance. We find that the source for this phenomenon can be attributed to the lack of 3D frequency constraints and the usage of a 2D dilation filter. To address this problem, we introduce a 3D smoothing filter which constrains the size of the 3D Gaussian primitives based on the maximal sampling frequency induced by the input views, eliminating high-frequency artifacts when zooming in. Moreover, replacing 2D dilation with a 2D Mip filter, which simulates a 2D box filter, effectively mitigates aliasing and dilation issues. Our evaluation, including scenarios such a training on single-scale images and testing on multiple scales, validates the effectiveness of our approach.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16493v1.pdf)

#### Relightable 3D Gaussian: Real-time Point Cloud Relighting with BRDF Decomposition and Ray Tracing

**Authors**: Yao Yao, Li Zhang, Xun Cao, Hao Zhu, Youtian Lin, Chun Gu, Jian Gao

<details>
<summary><b>Abstract</b></summary>
We present a novel differentiable point-based rendering framework for material and lighting decomposition from multi-view images, enabling editing, ray-tracing, and real-time relighting of the 3D point cloud. Specifically, a 3D scene is represented as a set of relightable 3D Gaussian points, where each point is additionally associated with a normal direction, BRDF parameters, and incident lights from different directions. To achieve robust lighting estimation, we further divide incident lights of each point into global and local components, as well as view-dependent visibilities. The 3D scene is optimized through the 3D Gaussian Splatting technique while BRDF and lighting are decomposed by physically-based differentiable rendering. Moreover, we introduce an innovative point-based ray-tracing approach based on the bounding volume hierarchy for efficient visibility baking, enabling real-time rendering and relighting of 3D Gaussian points with accurate shadow effects. Extensive experiments demonstrate improved BRDF estimation and novel view rendering results compared to state-of-the-art material estimation approaches. Our framework showcases the potential to revolutionize the mesh-based graphics pipeline with a relightable, traceable, and editable rendering pipeline solely based on point cloud. Project page:https://nju-3dv.github.io/projects/Relightable3DGaussian/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16043v1.pdf)

#### GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces

**Authors**: Yuexin Ma, Wenping Wang, Xiaoxiao Long, Xifeng Gao, YuAn Liu, Jiadong Tu, Yingwenqi Jiang

<details>
<summary><b>Abstract</b></summary>
The advent of neural 3D Gaussians has recently brought about a revolution in the field of neural rendering, facilitating the generation of high-quality renderings at real-time speeds. However, the explicit and discrete representation encounters challenges when applied to scenes featuring reflective surfaces. In this paper, we present GaussianShader, a novel method that applies a simplified shading function on 3D Gaussians to enhance the neural rendering in scenes with reflective surfaces while preserving the training and rendering efficiency. The main challenge in applying the shading function lies in the accurate normal estimation on discrete 3D Gaussians. Specifically, we proposed a novel normal estimation framework based on the shortest axis directions of 3D Gaussians with a delicately designed loss to make the consistency between the normals and the geometries of Gaussian spheres. Experiments show that GaussianShader strikes a commendable balance between efficiency and visual quality. Our method surpasses Gaussian Splatting in PSNR on specular object datasets, exhibiting an improvement of 1.57dB. When compared to prior works handling reflective surfaces, such as Ref-NeRF, our optimization time is significantly accelerated (23h vs. 0.58h). Please click on our project website to see more results.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17977v1.pdf)

#### Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering

**Authors**: Gim Hee Lee, Yu Chen, Weng Fei Low, Zhiwen Yan

<details>
<summary><b>Abstract</b></summary>
3D Gaussians have recently emerged as a highly efficient representation for 3D reconstruction and rendering. Despite its high rendering quality and speed at high resolutions, they both deteriorate drastically when rendered at lower resolutions or from far away camera position. During low resolution or far away rendering, the pixel size of the image can fall below the Nyquist frequency compared to the screen size of each splatted 3D Gaussian and leads to aliasing effect. The rendering is also drastically slowed down by the sequential alpha blending of more splatted Gaussians per pixel. To address these issues, we propose a multi-scale 3D Gaussian splatting algorithm, which maintains Gaussians at different scales to represent the same scene. Higher-resolution images are rendered with more small Gaussians, and lower-resolution images are rendered with fewer larger Gaussians. With similar training time, our algorithm can achieve 13\%-66\% PSNR and 160\%-2400\% rendering speed improvement at 4$\times$-128$\times$ scale rendering on Mip-NeRF360 dataset compared to the single scale 3D Gaussian splitting. Our code and more results are available on our project website https://jokeryan.github.io/projects/ms-gs/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17089v2.pdf)

#### Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting

**Authors**: Xiaogang Jin, Xiaojuan Qi, Shaohui Jiao, Wen Zhou, Xiaoyang Lyu, Yihua Huang, Yangtian Sun, Xinyu Gao, ZiYi Yang

<details>
<summary><b>Abstract</b></summary>
The recent advancements in 3D Gaussian splatting (3D-GS) have not only facilitated real-time rendering through modern GPU rasterization pipelines but have also attained state-of-the-art rendering quality. Nevertheless, despite its exceptional rendering quality and performance on standard datasets, 3D-GS frequently encounters difficulties in accurately modeling specular and anisotropic components. This issue stems from the limited ability of spherical harmonics (SH) to represent high-frequency information. To overcome this challenge, we introduce Spec-Gaussian, an approach that utilizes an anisotropic spherical Gaussian (ASG) appearance field instead of SH for modeling the view-dependent appearance of each 3D Gaussian. Additionally, we have developed a coarse-to-fine training strategy to improve learning efficiency and eliminate floaters caused by overfitting in real-world scenes. Our experimental results demonstrate that our method surpasses existing approaches in terms of rendering quality. Thanks to ASG, we have significantly improved the ability of 3D-GS to model scenes with specular and anisotropic components without increasing the number of 3D Gaussians. This improvement extends the applicability of 3D GS to handle intricate scenarios with specular and anisotropic surfaces.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.15870v1.pdf)

#### FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization

**Authors**: Eric Xing, Shijian Lu, Muyu Xu, Fangneng Zhan, Jiahui Zhang

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting has achieved very impressive performance in real-time novel view synthesis. However, it often suffers from over-reconstruction during Gaussian densification where high-variance image regions are covered by a few large Gaussians only, leading to blur and artifacts in the rendered images. We design a progressive frequency regularization (FreGS) technique to tackle the over-reconstruction issue within the frequency space. Specifically, FreGS performs coarse-to-fine Gaussian densification by exploiting low-to-high frequency components that can be easily extracted with low-pass and high-pass filters in the Fourier space. By minimizing the discrepancy between the frequency spectrum of the rendered image and the corresponding ground truth, it achieves high-quality Gaussian densification and alleviates the over-reconstruction of Gaussian splatting effectively. Experiments over multiple widely adopted benchmarks (e.g., Mip-NeRF360, Tanks-and-Temples and Deep Blending) show that FreGS achieves superior novel view synthesis and outperforms the state-of-the-art consistently.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.06908v2.pdf)

#### DeblurGS: Gaussian Splatting for Camera Motion Blur

**Authors**: Kyoung Mu Lee, Dongwoo Lee, JaeYoung Chung, Jeongtaek Oh

<details>
<summary><b>Abstract</b></summary>
Although significant progress has been made in reconstructing sharp 3D scenes from motion-blurred images, a transition to real-world applications remains challenging. The primary obstacle stems from the severe blur which leads to inaccuracies in the acquisition of initial camera poses through Structure-from-Motion, a critical aspect often overlooked by previous approaches. To address this challenge, we propose DeblurGS, a method to optimize sharp 3D Gaussian Splatting from motion-blurred images, even with the noisy camera pose initialization. We restore a fine-grained sharp scene by leveraging the remarkable reconstruction capability of 3D Gaussian Splatting. Our approach estimates the 6-Degree-of-Freedom camera motion for each blurry observation and synthesizes corresponding blurry renderings for the optimization process. Furthermore, we propose Gaussian Densification Annealing strategy to prevent the generation of inaccurate Gaussians at erroneous locations during the early training stages when camera motion is still imprecise. Comprehensive experiments demonstrate that our DeblurGS achieves state-of-the-art performance in deblurring and novel view synthesis for real-world and synthetic benchmark datasets, as well as field-captured blurry smartphone videos.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.11358v2.pdf)

#### GaussianPro: 3D Gaussian Splatting with Progressive Propagation

**Authors**: Xuejin Chen, Wenping Wang, Yuexin Ma, Wei Yin, Yao Yao, Kaizhi Yang, Xiaoxiao Long, Kai Cheng

<details>
<summary><b>Abstract</b></summary>
The advent of 3D Gaussian Splatting (3DGS) has recently brought about a revolution in the field of neural rendering, facilitating high-quality renderings at real-time speed. However, 3DGS heavily depends on the initialized point cloud produced by Structure-from-Motion (SfM) techniques. When tackling with large-scale scenes that unavoidably contain texture-less surfaces, the SfM techniques always fail to produce enough points in these surfaces and cannot provide good initialization for 3DGS. As a result, 3DGS suffers from difficult optimization and low-quality renderings. In this paper, inspired by classical multi-view stereo (MVS) techniques, we propose GaussianPro, a novel method that applies a progressive propagation strategy to guide the densification of the 3D Gaussians. Compared to the simple split and clone strategies used in 3DGS, our method leverages the priors of the existing reconstructed geometries of the scene and patch matching techniques to produce new Gaussians with accurate positions and orientations. Experiments on both large-scale and small-scale scenes validate the effectiveness of our method, where our method significantly surpasses 3DGS on the Waymo dataset, exhibiting an improvement of 1.15dB in terms of PSNR.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.14650v1.pdf)

#### SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing

**Authors**: Hao Zhao, Weihao Gu, Xiang He, Jingwei Zhao, Huan-ang Gao, Shiran Yuan, Jv Zheng, Xiaowei Song

<details>
<summary><b>Abstract</b></summary>
In this paper, we present a Scale-adaptive method for Anti-aliasing Gaussian Splatting (SA-GS). While the state-of-the-art method Mip-Splatting needs modifying the training procedure of Gaussian splatting, our method functions at test-time and is training-free. Specifically, SA-GS can be applied to any pretrained Gaussian splatting field as a plugin to significantly improve the field's anti-alising performance. The core technique is to apply 2D scale-adaptive filters to each Gaussian during test time. As pointed out by Mip-Splatting, observing Gaussians at different frequencies leads to mismatches between the Gaussian scales during training and testing. Mip-Splatting resolves this issue using 3D smoothing and 2D Mip filters, which are unfortunately not aware of testing frequency. In this work, we show that a 2D scale-adaptive filter that is informed of testing frequency can effectively match the Gaussian scale, thus making the Gaussian primitive distribution remain consistent across different testing frequencies. When scale inconsistency is eliminated, sampling rates smaller than the scene frequency result in conventional jaggedness, and we propose to integrate the projected 2D Gaussian within each pixel during testing. This integration is actually a limiting case of super-sampling, which significantly improves anti-aliasing performance over vanilla Gaussian Splatting. Through extensive experiments using various settings and both bounded and unbounded scenes, we show SA-GS performs comparably with or better than Mip-Splatting. Note that super-sampling and integration are only effective when our scale-adaptive filtering is activated. Our codes, data and models are available at https://github.com/zsy1987/SA-GS.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.19615v1.pdf)

## Generalization and Sparse Views
### Generalizable 3D Gaussian Splatting
#### AGG: Amortized Generative 3D Gaussians for Single Image to 3D

**Authors**: Arash Vahdat, Zhangyang Wang, Jiaming Song, Sifei Liu, Morteza Mardani, Ye Yuan, Dejia Xu

<details>
<summary><b>Abstract</b></summary>
Given the growing need for automatic 3D content creation pipelines, various 3D representations have been studied to generate 3D objects from a single image. Due to its superior rendering efficiency, 3D Gaussian splatting-based models have recently excelled in both 3D reconstruction and generation. 3D Gaussian splatting approaches for image to 3D generation are often optimization-based, requiring many computationally expensive score-distillation steps. To overcome these challenges, we introduce an Amortized Generative 3D Gaussian framework (AGG) that instantly produces 3D Gaussians from a single image, eliminating the need for per-instance optimization. Utilizing an intermediate hybrid representation, AGG decomposes the generation of 3D Gaussian locations and other appearance attributes for joint optimization. Moreover, we propose a cascaded pipeline that first generates a coarse representation of the 3D data and later upsamples it with a 3D Gaussian super-resolution module. Our method is evaluated against existing optimization-based 3D Gaussian frameworks and sampling-based pipelines utilizing other 3D representations, where AGG showcases competitive generation abilities both qualitatively and quantitatively while being several orders of magnitude faster. Project page: https://ir1d.github.io/AGG/
</details>

[📄 Paper](https://arxiv.org/pdf/2401.04099v1.pdf)

#### Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

**Authors**: Song-Hai Zhang, Yan-Pei Cao, Ding Liang, Yangguang Li, Yuan-Chen Guo, Zhipeng Yu, Zi-Xin Zou

<details>
<summary><b>Abstract</b></summary>
Recent advancements in 3D reconstruction from single images have been driven by the evolution of generative models. Prominent among these are methods based on Score Distillation Sampling (SDS) and the adaptation of diffusion models in the 3D domain. Despite their progress, these techniques often face limitations due to slow optimization or rendering processes, leading to extensive training and optimization times. In this paper, we introduce a novel approach for single-view reconstruction that efficiently generates a 3D model from a single image via feed-forward inference. Our method utilizes two transformer-based networks, namely a point decoder and a triplane decoder, to reconstruct 3D objects using a hybrid Triplane-Gaussian intermediate representation. This hybrid representation strikes a balance, achieving a faster rendering speed compared to implicit representations while simultaneously delivering superior rendering quality than explicit representations. The point decoder is designed for generating point clouds from single images, offering an explicit representation which is then utilized by the triplane decoder to query Gaussian features for each point. This design choice addresses the challenges associated with directly regressing explicit 3D Gaussian attributes characterized by their non-structural nature. Subsequently, the 3D Gaussians are decoded by an MLP to enable rapid rendering through splatting. Both decoders are built upon a scalable, transformer-based architecture and have been efficiently trained on large-scale 3D datasets. The evaluations conducted on both synthetic datasets and real-world images demonstrate that our method not only achieves higher quality but also ensures a faster runtime in comparison to previous state-of-the-art techniques. Please see our project page at https://zouzx.github.io/TriplaneGaussian/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.09147v2.pdf)

#### Splatter Image: Ultra-Fast Single-View 3D Reconstruction

**Authors**: Andrea Vedaldi, Christian Rupprecht, Stanislaw Szymanowicz

<details>
<summary><b>Abstract</b></summary>
We introduce the \method, an ultra-efficient approach for monocular 3D object reconstruction. Splatter Image is based on Gaussian Splatting, which allows fast and high-quality reconstruction of 3D scenes from multiple images. We apply Gaussian Splatting to monocular reconstruction by learning a neural network that, at test time, performs reconstruction in a feed-forward manner, at 38 FPS. Our main innovation is the surprisingly straightforward design of this network, which, using 2D operators, maps the input image to one 3D Gaussian per pixel. The resulting set of Gaussians thus has the form an image, the Splatter Image. We further extend the method take several images as input via cross-view attention. Owning to the speed of the renderer (588 FPS), we use a single GPU for training while generating entire images at each iteration to optimize perceptual metrics like LPIPS. On several synthetic, real, multi-category and large-scale benchmark datasets, we achieve better results in terms of PSNR, LPIPS, and other metrics while training and evaluating much faster than prior works. Code, models, demo and more results are available at https://szymanowiczs.github.io/splatter-image.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13150v2.pdf)

#### MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo

**Authors**: Hao Su, Jingyi Yu, Fanbo Xiang, Xiaoshuai Zhang, Fuqiang Zhao, Zexiang Xu, Anpei Chen

<details>
<summary><b>Abstract</b></summary>
We present MVSNeRF, a novel neural rendering approach that can efficiently reconstruct neural radiance fields for view synthesis. Unlike prior works on neural radiance fields that consider per-scene optimization on densely captured images, we propose a generic deep neural network that can reconstruct radiance fields from only three nearby input views via fast network inference. Our approach leverages plane-swept cost volumes (widely used in multi-view stereo) for geometry-aware scene reasoning, and combines this with physically based volume rendering for neural radiance field reconstruction. We train our network on real objects in the DTU dataset, and test it on three different datasets to evaluate its effectiveness and generalizability. Our approach can generalize across scenes (even indoor scenes, completely different from our training scenes of objects) and generate realistic view synthesis results using only three input images, significantly outperforming concurrent works on generalizable radiance field reconstruction. Moreover, if dense images are captured, our estimated radiance field representation can be easily fine-tuned; this leads to fast per-scene reconstruction with higher rendering quality and substantially less optimization time than NeRF.
</details>

[📄 Paper](https://arxiv.org/pdf/2103.15595v2.pdf)

#### pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

**Authors**: Vincent Sitzmann, Andrea Tagliasacchi, Sizhe Li, David Charatan

<details>
<summary><b>Abstract</b></summary>
We introduce pixelSplat, a feed-forward model that learns to reconstruct 3D radiance fields parameterized by 3D Gaussian primitives from pairs of images. Our model features real-time and memory-efficient rendering for scalable training as well as fast 3D reconstruction at inference time. To overcome local minima inherent to sparse and locally supported representations, we predict a dense probability distribution over 3D and sample Gaussian means from that probability distribution. We make this sampling operation differentiable via a reparameterization trick, allowing us to back-propagate gradients through the Gaussian splatting representation. We benchmark our method on wide-baseline novel view synthesis on the real-world RealEstate10k and ACID datasets, where we outperform state-of-the-art light field transformers and accelerate rendering by 2.5 orders of magnitude while reconstructing an interpretable and editable 3D radiance field.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.12337v4.pdf)

#### MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images

**Authors**: Jianfei Cai, Tat-Jen Cham, Andreas Geiger, Marc Pollefeys, Bohan Zhuang, Chuanxia Zheng, Haofei Xu, Yuedong Chen

<details>
<summary><b>Abstract</b></summary>
We introduce MVSplat, an efficient model that, given sparse multi-view images as input, predicts clean feed-forward 3D Gaussians. To accurately localize the Gaussian centers, we build a cost volume representation via plane sweeping, where the cross-view feature similarities stored in the cost volume can provide valuable geometry cues to the estimation of depth. We also learn other Gaussian primitives' parameters jointly with the Gaussian centers while only relying on photometric supervision. We demonstrate the importance of the cost volume representation in learning feed-forward Gaussians via extensive experimental evaluations. On the large-scale RealEstate10K and ACID benchmarks, MVSplat achieves state-of-the-art performance with the fastest feed-forward inference speed (22~fps). More impressively, compared to the latest state-of-the-art method pixelSplat, MVSplat uses $10\times$ fewer parameters and infers more than $2\times$ faster while providing higher appearance and geometry quality as well as better cross-dataset generalization.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.14627v2.pdf)

### Sparse Views Setting
#### Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting

**Authors**: Monroe Kennedy III, Mac Schwager, Gadiel Sznaier Camps, Won Kyung Do, Matthew Strong, Aiden Swann

<details>
<summary><b>Abstract</b></summary>
In this work, we propose a novel method to supervise 3D Gaussian Splatting (3DGS) scenes using optical tactile sensors. Optical tactile sensors have become widespread in their use in robotics for manipulation and object representation; however, raw optical tactile sensor data is unsuitable to directly supervise a 3DGS scene. Our representation leverages a Gaussian Process Implicit Surface to implicitly represent the object, combining many touches into a unified representation with uncertainty. We merge this model with a monocular depth estimation network, which is aligned in a two stage process, coarsely aligning with a depth camera and then finely adjusting to match our touch data. For every training image, our method produces a corresponding fused depth and uncertainty map. Utilizing this additional information, we propose a new loss function, variance weighted depth supervised loss, for training the 3DGS scene model. We leverage the DenseTact optical tactile sensor and RealSense RGB-D camera to show that combining touch and vision in this manner leads to quantitatively and qualitatively better results than vision or touch alone in a few-view scene syntheses on opaque as well as on reflective and transparent objects. Please see our project page at http://armlabstanford.github.io/touch-gs
</details>

[📄 Paper](https://arxiv.org/pdf/2403.09875v2.pdf)

#### DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization

**Authors**: Lin Gu, Jun Zhou, Xin Ning, Jin Zheng, Xiao Bai, Jiawei Zhang, Jiahe Li

<details>
<summary><b>Abstract</b></summary>
Radiance fields have demonstrated impressive performance in synthesizing novel views from sparse input views, yet prevailing methods suffer from high training costs and slow inference speed. This paper introduces DNGaussian, a depth-regularized framework based on 3D Gaussian radiance fields, offering real-time and high-quality few-shot novel view synthesis at low costs. Our motivation stems from the highly efficient representation and surprising quality of the recent 3D Gaussian Splatting, despite it will encounter a geometry degradation when input views decrease. In the Gaussian radiance fields, we find this degradation in scene geometry primarily lined to the positioning of Gaussian primitives and can be mitigated by depth constraint. Consequently, we propose a Hard and Soft Depth Regularization to restore accurate scene geometry under coarse monocular depth supervision while maintaining a fine-grained color appearance. To further refine detailed geometry reshaping, we introduce Global-Local Depth Normalization, enhancing the focus on small local depth changes. Extensive experiments on LLFF, DTU, and Blender datasets demonstrate that DNGaussian outperforms state-of-the-art methods, achieving comparable or better results with significantly reduced memory cost, a $25 \times$ reduction in training time, and over $3000 \times$ faster rendering speed.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.06912v3.pdf)

#### GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting

**Authors**: Qi Tian, Wei Shen, Xiaopeng Zhang, Lingxi Xie, Ruofan Liang, Jiemin Fang, Sikuang Li, Chen Yang

<details>
<summary><b>Abstract</b></summary>
Reconstructing and rendering 3D objects from highly sparse views is of critical importance for promoting applications of 3D vision techniques and improving user experience. However, images from sparse views only contain very limited 3D information, leading to two significant challenges: 1) Difficulty in building multi-view consistency as images for matching are too few; 2) Partially omitted or highly compressed object information as view coverage is insufficient. To tackle these challenges, we propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting, that achieves high rendering quality with only 4 input images. We first introduce techniques of visual hull and floater elimination which explicitly inject structure priors into the initial optimization process for helping build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. Our GaussianObject is evaluated on several challenging datasets, including MipNeRF360, OmniObject3D, and OpenIllumination, achieving strong reconstruction results from only 4 views and significantly outperforming previous state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10259v2.pdf)

#### Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images

**Authors**: Kyoung Mu Lee, Jeongtaek Oh, JaeYoung Chung

<details>
<summary><b>Abstract</b></summary>
In this paper, we present a method to optimize Gaussian splatting with a limited number of images while avoiding overfitting. Representing a 3D scene by combining numerous Gaussian splats has yielded outstanding visual quality. However, it tends to overfit the training views when only a small number of images are available. To address this issue, we introduce a dense depth map as a geometry guide to mitigate overfitting. We obtained the depth map using a pre-trained monocular depth estimation model and aligning the scale and offset using sparse COLMAP feature points. The adjusted depth aids in the color-based optimization of 3D Gaussian splatting, mitigating floating artifacts, and ensuring adherence to geometric constraints. We verify the proposed method on the NeRF-LLFF dataset with varying numbers of few images. Our approach demonstrates robust geometry compared to the original method that relies solely on images. Project page: robot0321.github.io/DepthRegGS
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13398v3.pdf)

#### FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

**Authors**: Zhangyang Wang, Yifan Jiang, Zhiwen Fan, Zehao Zhu

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender. Project website: https://zehaozhu.github.io/FSGS/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00451v2.pdf)

# Applications of 3D Gaussian Splatting
## Human Reconstruction
### Body Reconstruction
#### ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering

**Authors**: Marc Habermann, Christian Theobalt, Adam Kortylewski, Heming Zhu, Haokai Pang

<details>
<summary><b>Abstract</b></summary>
Real-time rendering of photorealistic and controllable human avatars stands as a cornerstone in Computer Vision and Graphics. While recent advances in neural implicit rendering have unlocked unprecedented photorealism for digital avatars, real-time performance has mostly been demonstrated for static scenes only. To address this, we propose ASH, an animatable Gaussian splatting approach for photorealistic rendering of dynamic humans in real-time. We parameterize the clothed human as animatable 3D Gaussians, which can be efficiently splatted into image space to generate the final rendering. However, naively learning the Gaussian parameters in 3D space poses a severe challenge in terms of compute. Instead, we attach the Gaussians onto a deformable character model, and learn their parameters in 2D texture space, which allows leveraging efficient 2D convolutional architectures that easily scale with the required number of Gaussians. We benchmark ASH with competing methods on pose-controllable avatars, demonstrating that our method outperforms existing real-time methods by a large margin and shows comparable or even better results than offline methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.05941v2.pdf)

#### Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling

**Authors**: Yebin Liu, Lizhen Wang, Zerong Zheng, Zhe Li

<details>
<summary><b>Abstract</b></summary>
    Modeling animatable human avatars from RGB videos is a long-standing and challenging problem. Recent works usually adopt MLP-based neural radiance fields (NeRF) to represent 3D humans but it remains difficult for pure MLPs to regress pose-dependent garment details. To this end we introduce Animatable Gaussians a new avatar representation that leverages powerful 2D CNNs and 3D Gaussian splatting to create high-fidelity avatars. To associate 3D Gaussians with the animatable avatar we learn a parametric template from the input videos and then parameterize the template on two front & back canonical Gaussian maps where each pixel represents a 3D Gaussian. The learned template is adaptive to the wearing garments for modeling looser clothes like dresses. Such template-guided 2D parameterization enables us to employ a powerful StyleGAN-based CNN to learn the pose-dependent Gaussian maps for modeling detailed dynamic appearances. Furthermore we introduce a pose projection strategy for better generalization given novel poses. Overall our method can create lifelike avatars with dynamic realistic and generalized appearances. Experiments show that our method outperforms other state-of-the-art approaches. Code: https://github.com/lizhe00/AnimatableGaussians.    
</details>

[📄 Paper](http://openaccess.thecvf.com//content/CVPR2024/papers/Li_Animatable_Gaussians_Learning_Pose-dependent_Gaussian_Maps_for_High-fidelity_Human_Avatar_CVPR_2024_paper.pdf)

#### GauHuman: Articulated Gaussian Splatting from Monocular Human Videos

**Authors**: Ziwei Liu, Shoukang Hu

<details>
<summary><b>Abstract</b></summary>
We present, GauHuman, a 3D human model with Gaussian Splatting for both fast training (1 ~ 2 minutes) and real-time rendering (up to 189 FPS), compared with existing NeRF-based implicit representation modelling frameworks demanding hours of training and seconds of rendering per frame. Specifically, GauHuman encodes Gaussian Splatting in the canonical space and transforms 3D Gaussians from canonical space to posed space with linear blend skinning (LBS), in which effective pose and LBS refinement modules are designed to learn fine details of 3D humans under negligible computational cost. Moreover, to enable fast optimization of GauHuman, we initialize and prune 3D Gaussians with 3D human prior, while splitting/cloning via KL divergence guidance, along with a novel merge operation for further speeding up. Extensive experiments on ZJU_Mocap and MonoCap datasets demonstrate that GauHuman achieves state-of-the-art performance quantitatively and qualitatively with fast training and real-time rendering speed. Notably, without sacrificing rendering quality, GauHuman can fast model the 3D human performer with ~13k 3D Gaussians.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02973v1.pdf)

#### GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians

**Authors**: Liqiang Nie, Shengping Zhang, Boning Liu, Boyao Zhou, Yuxiang Zhang, Hongwen Zhang, Liangxiao Hu

<details>
<summary><b>Abstract</b></summary>
We present GaussianAvatar, an efficient approach to creating realistic human avatars with dynamic 3D appearances from a single video. We start by introducing animatable 3D Gaussians to explicitly represent humans in various poses and clothing styles. Such an explicit and animatable representation can fuse 3D appearances more efficiently and consistently from 2D observations. Our representation is further augmented with dynamic properties to support pose-dependent appearance modeling, where a dynamic appearance network along with an optimizable feature tensor is designed to learn the motion-to-appearance mapping. Moreover, by leveraging the differentiable motion condition, our method enables a joint optimization of motions and appearances during avatar modeling, which helps to tackle the long-standing issue of inaccurate motion estimation in monocular settings. The efficacy of GaussianAvatar is validated on both the public dataset and our collected dataset, demonstrating its superior performances in terms of appearance quality and rendering efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02134v3.pdf)

#### 3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting

**Authors**: Siyu Tang, Andreas Geiger, Marko Mihajlovic, Shaofei Wang, Zhiyin Qian

<details>
<summary><b>Abstract</b></summary>
We introduce an approach that creates animatable human avatars from monocular videos using 3D Gaussian Splatting (3DGS). Existing methods based on neural radiance fields (NeRFs) achieve high-quality novel-view/novel-pose image synthesis but often require days of training, and are extremely slow at inference time. Recently, the community has explored fast grid structures for efficient training of clothed avatars. Albeit being extremely fast at training, these methods can barely achieve an interactive rendering frame rate with around 15 FPS. In this paper, we use 3D Gaussian Splatting and learn a non-rigid deformation network to reconstruct animatable clothed human avatars that can be trained within 30 minutes and rendered at real-time frame rates (50+ FPS). Given the explicit nature of our representation, we further introduce as-isometric-as-possible regularizations on both the Gaussian mean vectors and the covariance matrices, enhancing the generalization of our model on highly articulated unseen poses. Experimental results show that our method achieves comparable and even better performance compared to state-of-the-art approaches on animatable avatar creation from a monocular input, while being 400x and 250x faster in training and inference, respectively.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.09228v3.pdf)

#### Human Gaussian Splatting: Real-time Rendering of Animatable Avatars

**Authors**: Eduardo Pérez-Pellitero, Yiren Zhou, Richard Shaw, Helisa Dhamo, Jifei Song, Arthur Moreau

<details>
<summary><b>Abstract</b></summary>
This work addresses the problem of real-time rendering of photorealistic human body avatars learned from multi-view videos. While the classical approaches to model and render virtual humans generally use a textured mesh, recent research has developed neural body representations that achieve impressive visual quality. However, these models are difficult to render in real-time and their quality degrades when the character is animated with body poses different than the training observations. We propose an animatable human model based on 3D Gaussian Splatting, that has recently emerged as a very efficient alternative to neural radiance fields. The body is represented by a set of gaussian primitives in a canonical space which is deformed with a coarse to fine approach that combines forward skinning and local non-rigid refinement. We describe how to learn our Human Gaussian Splatting (HuGS) model in an end-to-end fashion from multi-view observations, and evaluate it against the state-of-the-art approaches for novel pose synthesis of clothed body. Our method achieves 1.5 dB PSNR improvement over the state-of-the-art on THuman4 dataset while being able to render in real-time (80 fps for 512x512 resolution).
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17113v2.pdf)

#### HUGS: Human Gaussian Splats

**Authors**: Anurag Ranjan, Oncel Tuzel, James Gabriel, Jen-Hao Rick Chang, Muhammed Kocabas

<details>
<summary><b>Abstract</b></summary>
Recent advances in neural rendering have improved both training and rendering times by orders of magnitude. While these methods demonstrate state-of-the-art quality and speed, they are designed for photogrammetry of static scenes and do not generalize well to freely moving humans in the environment. In this work, we introduce Human Gaussian Splats (HUGS) that represents an animatable human together with the scene using 3D Gaussian Splatting (3DGS). Our method takes only a monocular video with a small number of (50-100) frames, and it automatically learns to disentangle the static scene and a fully animatable human avatar within 30 minutes. We utilize the SMPL body model to initialize the human Gaussians. To capture details that are not modeled by SMPL (e.g. cloth, hairs), we allow the 3D Gaussians to deviate from the human body model. Utilizing 3D Gaussians for animated humans brings new challenges, including the artifacts created when articulating the Gaussians. We propose to jointly optimize the linear blend skinning weights to coordinate the movements of individual Gaussians during animation. Our approach enables novel-pose synthesis of human and novel view synthesis of both the human and the scene. We achieve state-of-the-art rendering quality with a rendering speed of 60 FPS while being ~100x faster to train over previous work. Our code will be announced here: https://github.com/apple/ml-hugs
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17910v1.pdf)

### Head Reconstruction
#### GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians

**Authors**: Matthias Nießner, Simon Giebenhain, Davide Davoli, Liam Schoneveld, Tobias Kirschstein, Shenhan Qian

<details>
<summary><b>Abstract</b></summary>
We introduce GaussianAvatars, a new method to create photorealistic head avatars that are fully controllable in terms of expression, pose, and viewpoint. The core idea is a dynamic 3D representation based on 3D Gaussian splats that are rigged to a parametric morphable face model. This combination facilitates photorealistic rendering while allowing for precise animation control via the underlying parametric model, e.g., through expression transfer from a driving sequence or by manually changing the morphable model parameters. We parameterize each splat by a local coordinate frame of a triangle and optimize for explicit displacement offset to obtain a more accurate geometric representation. During avatar reconstruction, we jointly optimize for the morphable model parameters and Gaussian splat parameters in an end-to-end fashion. We demonstrate the animation capabilities of our photorealistic avatar in several challenging scenarios. For instance, we show reenactments from a driving video, where our method outperforms existing works by a significant margin.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02069v2.pdf)

#### Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians

**Authors**: Yebin Liu, Zerong Zheng, Lizhen Wang, Hongwen Zhang, Zhe Li, Benwang Chen, Yuelang Xu

<details>
<summary><b>Abstract</b></summary>
Creating high-fidelity 3D head avatars has always been a research hotspot, but there remains a great challenge under lightweight sparse view setups. In this paper, we propose Gaussian Head Avatar represented by controllable 3D Gaussians for high-fidelity head avatar modeling. We optimize the neutral 3D Gaussians and a fully learned MLP-based deformation field to capture complex expressions. The two parts benefit each other, thereby our method can model fine-grained dynamic details while ensuring expression accuracy. Furthermore, we devise a well-designed geometry-guided initialization strategy based on implicit SDF and Deep Marching Tetrahedra for the stability and convergence of the training procedure. Experiments show our approach outperforms other state-of-the-art sparse-view methods, achieving ultra high-fidelity rendering quality at 2K resolution even under exaggerated expressions.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.03029v2.pdf)

### Others
#### GaussianHair: Hair Modeling and Rendering with Light-aware Gaussians

**Authors**: Jingyi Yu, Lan Xu, Wei Yang, Qixuan Zhang, Longwen Zhang, Suyi Jiang, Zijun Zhao, Min Ouyang, Haimin Luo

<details>
<summary><b>Abstract</b></summary>
Hairstyle reflects culture and ethnicity at first glance. In the digital era, various realistic human hairstyles are also critical to high-fidelity digital human assets for beauty and inclusivity. Yet, realistic hair modeling and real-time rendering for animation is a formidable challenge due to its sheer number of strands, complicated structures of geometry, and sophisticated interaction with light. This paper presents GaussianHair, a novel explicit hair representation. It enables comprehensive modeling of hair geometry and appearance from images, fostering innovative illumination effects and dynamic animation capabilities. At the heart of GaussianHair is the novel concept of representing each hair strand as a sequence of connected cylindrical 3D Gaussian primitives. This approach not only retains the hair's geometric structure and appearance but also allows for efficient rasterization onto a 2D image plane, facilitating differentiable volumetric rendering. We further enhance this model with the "GaussianHair Scattering Model", adept at recreating the slender structure of hair strands and accurately capturing their local diffuse color in uniform lighting. Through extensive experiments, we substantiate that GaussianHair achieves breakthroughs in both geometric and appearance fidelity, transcending the limitations encountered in state-of-the-art methods for hair reconstruction. Beyond representation, GaussianHair extends to support editing, relighting, and dynamic rendering of hair, offering seamless integration with conventional CG pipeline workflows. Complementing these advancements, we have compiled an extensive dataset of real human hair, each with meticulously detailed strand geometry, to propel further research in this field.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10483v1.pdf)

## Non-Rigid Object Reconstruction
#### GART: Gaussian Articulated Template Models

**Authors**: Kostas Daniilidis, Lingjie Liu, Georgios Pavlakos, Yufu Wang, Jiahui Lei

<details>
<summary><b>Abstract</b></summary>
We introduce Gaussian Articulated Template Model GART, an explicit, efficient, and expressive representation for non-rigid articulated subject capturing and rendering from monocular videos. GART utilizes a mixture of moving 3D Gaussians to explicitly approximate a deformable subject's geometry and appearance. It takes advantage of a categorical template model prior (SMPL, SMAL, etc.) with learnable forward skinning while further generalizing to more complex non-rigid deformations with novel latent bones. GART can be reconstructed via differentiable rendering from monocular videos in seconds or minutes and rendered in novel poses faster than 150fps.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16099v1.pdf)

#### Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction

**Authors**: Jan Eric Lenssen, Eddy Ilg, Raza Yunus, Christopher Wewer, Devikalyan Das

<details>
<summary><b>Abstract</b></summary>
Reconstructing dynamic objects from monocular videos is a severely underconstrained and challenging problem, and recent work has approached it in various directions. However, owing to the ill-posed nature of this problem, there has been no solution that can provide consistent, high-quality novel views from camera positions that are significantly different from the training views. In this work, we introduce Neural Parametric Gaussians (NPGs) to take on this challenge by imposing a two-stage approach: first, we fit a low-rank neural deformation model, which then is used as regularization for non-rigid reconstruction in the second stage. The first stage learns the object's deformations such that it preserves consistency in novel views. The second stage obtains high reconstruction quality by optimizing 3D Gaussians that are driven by the coarse model. To this end, we introduce a local 3D Gaussian representation, where temporally shared Gaussians are anchored in and deformed by local oriented volumes. The resulting combined model can be rendered as radiance fields, resulting in high-quality photo-realistic reconstructions of the non-rigidly deforming objects. We demonstrate that NPGs achieve superior results compared to previous works, especially in challenging scenarios with few multi-view cues.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.01196v2.pdf)

## Artificial Intelligence-Generated Content (AIGC)
### Text to 3D Objects
#### Hyper-3DG: Text-to-3D Gaussian Generation via Hypergraph

**Authors**: Yue Gao, Xun Yang, Wei Chen, Zhou Xue, Chaofan Luo, Jiahui Yang, Donglin Di

<details>
<summary><b>Abstract</b></summary>
Text-to-3D generation represents an exciting field that has seen rapid advancements, facilitating the transformation of textual descriptions into detailed 3D models. However, current progress often neglects the intricate high-order correlation of geometry and texture within 3D objects, leading to challenges such as over-smoothness, over-saturation and the Janus problem. In this work, we propose a method named ``3D Gaussian Generation via Hypergraph (Hyper-3DG)'', designed to capture the sophisticated high-order correlations present within 3D objects. Our framework is anchored by a well-established mainflow and an essential module, named ``Geometry and Texture Hypergraph Refiner (HGRefiner)''. This module not only refines the representation of 3D Gaussians but also accelerates the update process of these 3D Gaussians by conducting the Patch-3DGS Hypergraph Learning on both explicit attributes and latent visual features. Our framework allows for the production of finely generated 3D objects within a cohesive optimization, effectively circumventing degradation. Extensive experimentation has shown that our proposed method significantly enhances the quality of 3D generation while incurring no additional computational overhead for the underlying framework. (Project code: https://github.com/yjhboy/Hyper3DG)
</details>
[📄 Paper](https://arxiv.org/pdf/2403.09236v1.pdf)

#### Text-to-3D using Gaussian Splatting

**Authors**: Huaping Liu, Yikai Wang, Feng Wang, Zilong Chen

<details>
<summary><b>Abstract</b></summary>
Automatic text-to-3D generation that combines Score Distillation Sampling (SDS) with the optimization of volume rendering has achieved remarkable progress in synthesizing realistic 3D objects. Yet most existing text-to-3D methods by SDS and volume rendering suffer from inaccurate geometry, e.g., the Janus issue, since it is hard to explicitly integrate 3D priors into implicit 3D representations. Besides, it is usually time-consuming for them to generate elaborate 3D models with rich colors. In response, this paper proposes GSGEN, a novel method that adopts Gaussian Splatting, a recent state-of-the-art representation, to text-to-3D generation. GSGEN aims at generating high-quality 3D objects and addressing existing shortcomings by exploiting the explicit nature of Gaussian Splatting that enables the incorporation of 3D prior. Specifically, our method adopts a progressive optimization strategy, which includes a geometry optimization stage and an appearance refinement stage. In geometry optimization, a coarse representation is established under 3D point cloud diffusion prior along with the ordinary 2D SDS optimization, ensuring a sensible and 3D-consistent rough shape. Subsequently, the obtained Gaussians undergo an iterative appearance refinement to enrich texture details. In this stage, we increase the number of Gaussians by compactness-based densification to enhance continuity and improve fidelity. With these designs, our approach can generate 3D assets with delicate details and accurate geometry. Extensive evaluations demonstrate the effectiveness of our method, especially for capturing high-frequency components. Our code is available at https://github.com/gsgen3d/gsgen
</details>
[📄 Paper](https://arxiv.org/pdf/2309.16585v4.pdf)

#### GVGEN: Text-to-3D Generation with Volumetric Representation

**Authors**: Tong He, Wanli Ouyang, Chun Yuan, Xiaoshui Huang, Yangguang Li, Di Huang, Sida Peng, Junyi Chen, Xianglong He

<details>
<summary><b>Abstract</b></summary>
In recent years, 3D Gaussian splatting has emerged as a powerful technique for 3D reconstruction and generation, known for its fast and high-quality rendering capabilities. To address these shortcomings, this paper introduces a novel diffusion-based framework, GVGEN, designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques:(1) Structured Volumetric Representation. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. To better optimize the representation of these details, we propose a unique pruning and densifying method named the Candidate Pool Strategy, enhancing detail fidelity through selective optimization. (2) Coarse-to-fine Generation Pipeline. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. Our framework, GVGEN, demonstrates superior performance in qualitative and quantitative assessments compared to existing 3D generation methods. Simultaneously, it maintains a fast generation speed ($\sim$7 seconds), effectively striking a balance between quality and efficiency. Our project page is: https://gvgen.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12957v2.pdf)

#### BrightDreamer: Generic 3D Gaussian Generative Framework for Fast Text-to-3D Synthesis

**Authors**: Lin Wang, Lutao Jiang

<details>
<summary><b>Abstract</b></summary>
Text-to-3D synthesis has recently seen intriguing advances by combining the text-to-image models with 3D representation methods, e.g., Gaussian Splatting (GS), via Score Distillation Sampling (SDS). However, a hurdle of existing methods is the low efficiency, per-prompt optimization for a single 3D object. Therefore, it is imperative for a paradigm shift from per-prompt optimization to one-stage generation for any unseen text prompts, which yet remains challenging. A hurdle is how to directly generate a set of millions of 3D Gaussians to represent a 3D object. This paper presents BrightDreamer, an end-to-end single-stage approach that can achieve generalizable and fast (77 ms) text-to-3D generation. Our key idea is to formulate the generation process as estimating the 3D deformation from an anchor shape with predefined positions. For this, we first propose a Text-guided Shape Deformation (TSD) network to predict the deformed shape and its new positions, used as the centers (one attribute) of 3D Gaussians. To estimate the other four attributes (i.e., scaling, rotation, opacity, and SH coefficient), we then design a novel Text-guided Triplane Generator (TTG) to generate a triplane representation for a 3D object. The center of each Gaussian enables us to transform the triplane feature into the four attributes. The generated 3D Gaussians can be finally rendered at 705 frames per second. Extensive experiments demonstrate the superiority of our method over existing methods. Also, BrightDreamer possesses a strong semantic understanding capability even for complex text prompts. The project code is available at https://vlislab22.github.io/BrightDreamer.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11273v1.pdf)

#### Learn to Optimize Denoising Scores for 3D Generation: A Unified and Improved Diffusion Prior on NeRF and 3D Gaussian Splatting

**Authors**: Guosheng Lin, Fayao Liu, Xulei Yang, Yi Xu, Chi Zhang, Cheng Chen, YiWen Chen, Xiaofeng Yang

<details>
<summary><b>Abstract</b></summary>
We propose a unified framework aimed at enhancing the diffusion priors for 3D generation tasks. Despite the critical importance of these tasks, existing methodologies often struggle to generate high-caliber results. We begin by examining the inherent limitations in previous diffusion priors. We identify a divergence between the diffusion priors and the training procedures of diffusion models that substantially impairs the quality of 3D generation. To address this issue, we propose a novel, unified framework that iteratively optimizes both the 3D model and the diffusion prior. Leveraging the different learnable parameters of the diffusion prior, our approach offers multiple configurations, affording various trade-offs between performance and implementation complexity. Notably, our experimental results demonstrate that our method markedly surpasses existing techniques, establishing new state-of-the-art in the realm of text-to-3D generation. Furthermore, our approach exhibits impressive performance on both NeRF and the newly introduced 3D Gaussian Splatting backbones. Additionally, our framework yields insightful contributions to the understanding of recent score distillation methods, such as the VSD and DDS loss.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.04820v1.pdf)

#### IM-3D: Iterative Multiview Diffusion and Reconstruction for High-Quality 3D Generation

**Authors**: Filippos Kokkinos, Oran Gafni, Andrea Vedaldi, Natalia Neverova, Christian Rupprecht, Iro Laina, Luke Melas-Kyriazi

<details>
<summary><b>Abstract</b></summary>
Most text-to-3D generators build upon off-the-shelf text-to-image models trained on billions of images. They use variants of Score Distillation Sampling (SDS), which is slow, somewhat unstable, and prone to artifacts. A mitigation is to fine-tune the 2D generator to be multi-view aware, which can help distillation or can be combined with reconstruction networks to output 3D objects directly. In this paper, we further explore the design space of text-to-3D models. We significantly improve multi-view generation by considering video instead of image generators. Combined with a 3D reconstruction algorithm which, by using Gaussian splatting, can optimize a robust image-based loss, we directly produce high-quality 3D outputs from the generated views. Our new method, IM-3D, reduces the number of evaluations of the 2D generator network 10-100x, resulting in a much more efficient pipeline, better quality, fewer geometric inconsistencies, and higher yield of usable 3D assets.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.08682v1.pdf)

#### HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

**Authors**: Ziwei Liu, Xihui Liu, Dahua Lin, Gang Zeng, Ying Shan, Jiaxiang Tang, Xiaohang Zhan, Xian Liu

<details>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17061v2.pdf)

#### GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise

**Authors**: Kuo-Kun Tseng, Huaibin Wang, Xinhai Li

<details>
<summary><b>Abstract</b></summary>
Text-to-3D, known for its efficient generation methods and expansive creative potential, has garnered significant attention in the AIGC domain. However, the amalgamation of Nerf and 2D diffusion models frequently yields oversaturated images, posing severe limitations on downstream industrial applications due to the constraints of pixelwise rendering method. Gaussian splatting has recently superseded the traditional pointwise sampling technique prevalent in NeRF-based methodologies, revolutionizing various aspects of 3D reconstruction. This paper introduces a novel text to 3D content generation framework based on Gaussian splatting, enabling fine control over image saturation through individual Gaussian sphere transparencies, thereby producing more realistic images. The challenge of achieving multi-view consistency in 3D generation significantly impedes modeling complexity and accuracy. Taking inspiration from SJC, we explore employing multi-view noise distributions to perturb images generated by 3D Gaussian splatting, aiming to rectify inconsistencies in multi-view geometry. We ingeniously devise an efficient method to generate noise that produces Gaussian noise from diverse viewpoints, all originating from a shared noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap models in local minima, causing artifacts like floaters, burrs, or proliferative elements. To mitigate these issues, we propose the variational Gaussian splatting technique to enhance the quality and stability of 3D appearance. To our knowledge, our approach represents the first comprehensive utilization of Gaussian splatting across the entire spectrum of 3D content generation processes.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11221v1.pdf)

#### Gaussian Shell Maps for Efficient 3D Human Generation

**Authors**: Gordon Wetzstein, Dit-yan Yeung, Qifeng Chen, Zhengfei Kuang, Ryan Po, Yinghao Xu, Zifan Shi, Wang Yifan, Rameen Abdal

<details>
<summary><b>Abstract</b></summary>
Efficient generation of 3D digital humans is important in several industries, including virtual reality, social media, and cinematic production. 3D generative adversarial networks (GANs) have demonstrated state-of-the-art (SOTA) quality and diversity for generated assets. Current 3D GAN architectures, however, typically rely on volume representations, which are slow to render, thereby hampering the GAN training and requiring multi-view-inconsistent 2D upsamplers. Here, we introduce Gaussian Shell Maps (GSMs) as a framework that connects SOTA generator network architectures with emerging 3D Gaussian rendering primitives using an articulable multi shell--based scaffold. In this setting, a CNN generates a 3D texture stack with features that are mapped to the shells. The latter represent inflated and deflated versions of a template surface of a digital human in a canonical body pose. Instead of rasterizing the shells directly, we sample 3D Gaussians on the shells whose attributes are encoded in the texture features. These Gaussians are efficiently and differentiably rendered. The ability to articulate the shells is important during GAN training and, at inference time, to deform a body into arbitrary user-defined poses. Our efficient rendering scheme bypasses the need for view-inconsistent upsamplers and achieves high-quality multi-view consistent renderings at a native resolution of $512 \times 512$ pixels. We demonstrate that GSMs successfully generate 3D humans when trained on single-view datasets, including SHHQ and DeepFashion.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17857v1.pdf)

#### GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models

**Authors**: Lingxi Xie, Junjie Wang, Xinggang Wang, Qi Tian, Wenyu Liu, Xiaopeng Zhang, Guanjun Wu, Jiemin Fang, Taoran Yi

<details>
<summary><b>Abstract</b></summary>
In recent times, the generation of 3D assets from text prompts has shown impressive results. Both 2D and 3D diffusion models can help generate decent 3D objects based on prompts. 3D diffusion models have good 3D consistency, but their quality and generalization are limited as trainable 3D data is expensive and hard to obtain. 2D diffusion models enjoy strong abilities of generalization and fine generation, but 3D consistency is hard to guarantee. This paper attempts to bridge the power from the two types of diffusion models via the recent explicit and efficient 3D Gaussian splatting representation. A fast 3D object generation framework, named as GaussianDreamer, is proposed, where the 3D diffusion model provides priors for initialization and the 2D diffusion model enriches the geometry and appearance. Operations of noisy point growing and color perturbation are introduced to enhance the initialized Gaussians. Our GaussianDreamer can generate a high-quality 3D instance or 3D avatar within 15 minutes on one GPU, much faster than previous methods, while the generated instances can be directly rendered in real time. Demos and code are available at https://taoranyi.com/gaussiandreamer/.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.08529v3.pdf)

#### LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

**Authors**: Ziwei Liu, Gang Zeng, Tengfei Wang, Xiaokang Chen, Zhaoxi Chen, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
3D content creation has achieved significant progress in terms of both quality and speed. Although current feed-forward models can produce 3D objects in seconds, their resolution is constrained by the intensive computation required during training. In this paper, we introduce Large Multi-View Gaussian Model (LGM), a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models. Extensive experiments demonstrate the high fidelity and efficiency of our approach. Notably, we maintain the fast speed to generate 3D objects within 5 seconds while boosting the training resolution to 512, thereby achieving high-resolution 3D content generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.05054v1.pdf)

#### Gaussian Shell Maps for Efficient 3D Human Generation

**Authors**: Gordon Wetzstein, Dit-yan Yeung, Qifeng Chen, Zhengfei Kuang, Ryan Po, Yinghao Xu, Zifan Shi, Wang Yifan, Rameen Abdal

<details>
<summary><b>Abstract</b></summary>
Efficient generation of 3D digital humans is important in several industries, including virtual reality, social media, and cinematic production. 3D generative adversarial networks (GANs) have demonstrated state-of-the-art (SOTA) quality and diversity for generated assets. Current 3D GAN architectures, however, typically rely on volume representations, which are slow to render, thereby hampering the GAN training and requiring multi-view-inconsistent 2D upsamplers. Here, we introduce Gaussian Shell Maps (GSMs) as a framework that connects SOTA generator network architectures with emerging 3D Gaussian rendering primitives using an articulable multi shell--based scaffold. In this setting, a CNN generates a 3D texture stack with features that are mapped to the shells. The latter represent inflated and deflated versions of a template surface of a digital human in a canonical body pose. Instead of rasterizing the shells directly, we sample 3D Gaussians on the shells whose attributes are encoded in the texture features. These Gaussians are efficiently and differentiably rendered. The ability to articulate the shells is important during GAN training and, at inference time, to deform a body into arbitrary user-defined poses. Our efficient rendering scheme bypasses the need for view-inconsistent upsamplers and achieves high-quality multi-view consistent renderings at a native resolution of $512 \times 512$ pixels. We demonstrate that GSMs successfully generate 3D humans when trained on single-view datasets, including SHHQ and DeepFashion.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17857v1.pdf)

#### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

**Authors**: Gang Zeng, Ziwei Liu, Hang Zhou, Jiawei Ren, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
</details>
[📄 Paper](https://arxiv.org/pdf/2309.16653v2.pdf)

#### ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation

**Authors**: Jun Zhu, Hang Su, Chongxuan Li, Fan Bao, Yikai Wang, Cheng Lu, Zhengyi Wang

<details>
<summary><b>Abstract</b></summary>
Score distillation sampling (SDS) has shown great promise in text-to-3D generation by distilling pretrained large-scale text-to-image diffusion models, but suffers from over-saturation, over-smoothing, and low-diversity problems. In this work, we propose to model the 3D parameter as a random variable instead of a constant as in SDS and present variational score distillation (VSD), a principled particle-based variational framework to explain and address the aforementioned issues in text-to-3D generation. We show that SDS is a special case of VSD and leads to poor samples with both small and large CFG weights. In comparison, VSD works well with various CFG weights as ancestral sampling from diffusion models and simultaneously improves the diversity and sample quality with a common CFG weight (i.e., $7.5$). We further present various improvements in the design space for text-to-3D such as distillation time schedule and density initialization, which are orthogonal to the distillation algorithm yet not well explored. Our overall approach, dubbed ProlificDreamer, can generate high rendering resolution (i.e., $512\times512$) and high-fidelity NeRF with rich structure and complex effects (e.g., smoke and drops). Further, initialized from NeRF, meshes fine-tuned by VSD are meticulously detailed and photo-realistic. Project page and codes: https://ml.cs.tsinghua.edu.cn/prolificdreamer/
</details>

[📄 Paper](https://arxiv.org/pdf/2305.16213v2.pdf)

#### LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching

**Authors**: Yingcong Chen, Xiaogang Xu, Haodong Li, Jiantao Lin, Xin Yang, Yixun Liang

<details>
<summary><b>Abstract</b></summary>
The recent advancements in text-to-3D generation mark a significant milestone in generative models, unlocking new possibilities for creating imaginative 3D assets across various real-world scenarios. While recent advancements in text-to-3D generation have shown promise, they often fall short in rendering detailed and high-quality 3D models. This problem is especially prevalent as many methods base themselves on Score Distillation Sampling (SDS). This paper identifies a notable deficiency in SDS, that it brings inconsistent and low-quality updating direction for the 3D model, causing the over-smoothing effect. To address this, we propose a novel approach called Interval Score Matching (ISM). ISM employs deterministic diffusing trajectories and utilizes interval-based score matching to counteract over-smoothing. Furthermore, we incorporate 3D Gaussian Splatting into our text-to-3D generation pipeline. Extensive experiments show that our model largely outperforms the state-of-the-art in quality and training efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11284v3.pdf)

### Image to 3D Object
#### Zero-1-to-3: Zero-shot One Image to 3D Object

**Authors**: Carl Vondrick, Sergey Zakharov, Pavel Tokmakov, Basile Van Hoorick, Rundi Wu, Ruoshi Liu

<details>
<summary><b>Abstract</b></summary>
We introduce Zero-1-to-3, a framework for changing the camera viewpoint of an object given just a single RGB image. To perform novel view synthesis in this under-constrained setting, we capitalize on the geometric priors that large-scale diffusion models learn about natural images. Our conditional diffusion model uses a synthetic dataset to learn controls of the relative camera viewpoint, which allow new images to be generated of the same object under a specified camera transformation. Even though it is trained on a synthetic dataset, our model retains a strong zero-shot generalization ability to out-of-distribution datasets as well as in-the-wild images, including impressionist paintings. Our viewpoint-conditioned diffusion approach can further be used for the task of 3D reconstruction from a single image. Qualitative and quantitative experiments show that our method significantly outperforms state-of-the-art single-view 3D reconstruction and novel view synthesis models by leveraging Internet-scale pre-training.
</details>

[📄 Paper](https://arxiv.org/pdf/2303.11328v1.pdf)

#### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

**Authors**: Gang Zeng, Ziwei Liu, Hang Zhou, Jiawei Ren, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.16653v2.pdf)

#### FDGaussian: Fast Gaussian Splatting from Single Image via Geometric-aware Diffusion Model

**Authors**: Yu-Gang Jiang, Zuxuan Wu, Zhen Xing, Qijun Feng

<details>
<summary><b>Abstract</b></summary>
Reconstructing detailed 3D objects from single-view images remains a challenging task due to the limited information available. In this paper, we introduce FDGaussian, a novel two-stage framework for single-image 3D reconstruction. Recent methods typically utilize pre-trained 2D diffusion models to generate plausible novel views from the input image, yet they encounter issues with either multi-view inconsistency or lack of geometric fidelity. To overcome these challenges, we propose an orthogonal plane decomposition mechanism to extract 3D geometric features from the 2D input, enabling the generation of consistent multi-view images. Moreover, we further accelerate the state-of-the-art Gaussian Splatting incorporating epipolar attention to fuse images from different viewpoints. We demonstrate that FDGaussian generates images with high consistency across different views and reconstructs high-quality 3D objects, both qualitatively and quantitatively. More examples can be found at our website https://qjfeng.net/FDGaussian/.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.10242v1.pdf)

#### Repaint123: Fast and High-quality One Image to 3D Generation with Progressive Controllable 2D Repainting

**Authors**: Li Yuan, Munan Ning, Yida Wei, Peng Jin, Xinhua Cheng, Yatian Pang, Zhenyu Tang, Junwu Zhang

<details>
<summary><b>Abstract</b></summary>
Recent one image to 3D generation methods commonly adopt Score Distillation Sampling (SDS). Despite the impressive results, there are multiple deficiencies including multi-view inconsistency, over-saturated and over-smoothed textures, as well as the slow generation speed. To address these deficiencies, we present Repaint123 to alleviate multi-view bias as well as texture degradation and speed up the generation process. The core idea is to combine the powerful image generation capability of the 2D diffusion model and the texture alignment ability of the repainting strategy for generating high-quality multi-view images with consistency. We further propose visibility-aware adaptive repainting strength for overlap regions to enhance the generated image quality in the repainting process. The generated high-quality and multi-view consistent images enable the use of simple Mean Square Error (MSE) loss for fast 3D content generation. We conduct extensive experiments and show that our method has a superior ability to generate high-quality 3D content with multi-view consistency and fine textures in 2 minutes from scratch. Our project page is available at https://pku-yuangroup.github.io/repaint123/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13271v3.pdf)

### Multi-Object and Scene Generation
#### CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting

**Authors**: Achuta Kadambi, Pradyumna Chari, Alexander Vilesov

<details>
<summary><b>Abstract</b></summary>
With the onset of diffusion-based generative models and their ability to generate text-conditioned images, content generation has received a massive invigoration. Recently, these models have been shown to provide useful guidance for the generation of 3D graphics assets. However, existing work in text-conditioned 3D generation faces fundamental constraints: (i) inability to generate detailed, multi-object scenes, (ii) inability to textually control multi-object configurations, and (iii) physically realistic scene composition. In this work, we propose CG3D, a method for compositionally generating scalable 3D assets that resolves these constraints. We find that explicit Gaussian radiance fields, parameterized to allow for compositions of objects, possess the capability to enable semantically and physically consistent scenes. By utilizing a guidance framework built around this explicit representation, we show state of the art results, capable of even exceeding the guiding diffusion model in terms of object combinations and physics accuracy.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17907v1.pdf)

#### Alpha shapes: determining 3D shape complexity across morphologically diverse structures
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting

**Authors**: Ming-Hsuan Yang, Deqing Sun, Yongtao Wang, Zhiwei Lin, Jinlin He, Yajiao Xiong, Xingjian Ran, Xiaoyu Zhou

<details>
<summary><b>Abstract</b></summary>
We present GALA3D, generative 3D GAussians with LAyout-guided control, for effective compositional text-to-3D generation. We first utilize large language models (LLMs) to generate the initial layout and introduce a layout-guided 3D Gaussian representation for 3D content generation with adaptive geometric constraints. We then propose an instance-scene compositional optimization mechanism with conditioned diffusion to collaboratively generate realistic 3D scenes with consistent geometry, texture, scale, and accurate interactions among multiple objects while simultaneously adjusting the coarse layout priors extracted from the LLMs to align with the generated scene. Experiments show that GALA3D is a user-friendly, end-to-end framework for state-of-the-art scene-level 3D content generation and controllable editing while ensuring the high fidelity of object-level entities within the scene. The source codes and models will be available at gala3d.github.io.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.07207v2.pdf)

#### RePaint: Inpainting using Denoising Diffusion Probabilistic Models

**Authors**: Luc van Gool, Radu Timofte, Fisher Yu, Andres Romero, Martin Danelljan, Andreas Lugmayr

<details>
<summary><b>Abstract</b></summary>
Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image information. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. RePaint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions. Github Repository: git.io/RePaint
</details>

[📄 Paper](https://arxiv.org/pdf/2201.09865v4.pdf)

#### Text2Immersion: Generative Immersive Scene with 3D Gaussians

**Authors**: Tiancheng Sun, Stephen Lombardi, Kathryn Heal, Hao Ouyang

<details>
<summary><b>Abstract</b></summary>
We introduce Text2Immersion, an elegant method for producing high-quality 3D immersive scenes from text prompts. Our proposed pipeline initiates by progressively generating a Gaussian cloud using pre-trained 2D diffusion and depth estimation models. This is followed by a refining stage on the Gaussian cloud, interpolating and refining it to enhance the details of the generated scene. Distinct from prevalent methods that focus on single object or indoor scenes, or employ zoom-out trajectories, our approach generates diverse scenes with various objects, even extending to the creation of imaginary scenes. Consequently, Text2Immersion can have wide-ranging implications for various applications such as virtual reality, game development, and automated content creation. Extensive evaluations demonstrate that our system surpasses other methods in rendering quality and diversity, further progressing towards text-driven 3D scene generation. We will make the source code publicly accessible at the project page.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.09242v1.pdf)

#### LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes

**Authors**: Kyoung Mu Lee, Jaerin Lee, Hyeongjin Nam, Suyoung Lee, JaeYoung Chung

<details>
<summary><b>Abstract</b></summary>
With the widespread usage of VR devices and contents, demands for 3D scene generation techniques become more popular. Existing 3D scene generation models, however, limit the target scene to specific domain, primarily due to their training strategies using 3D scan dataset that is far from the real-world. To address such limitation, we propose LucidDreamer, a domain-free scene generation pipeline by fully leveraging the power of existing large-scale diffusion-based generative model. Our LucidDreamer has two alternate steps: Dreaming and Alignment. First, to generate multi-view consistent images from inputs, we set the point cloud as a geometrical guideline for each image generation. Specifically, we project a portion of point cloud to the desired view and provide the projection as a guidance for inpainting using the generative model. The inpainted images are lifted to 3D space with estimated depth maps, composing a new points. Second, to aggregate the new points into the 3D scene, we propose an aligning algorithm which harmoniously integrates the portions of newly generated 3D scenes. The finally obtained 3D scene serves as initial points for optimizing Gaussian splats. LucidDreamer produces Gaussian splats that are highly-detailed compared to the previous 3D scene generation methods, with no constraint on domain of the target scene. Project page: https://luciddreamer-cvlab.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13384v2.pdf)

### 4D Generation
#### GaussianFlow: Splatting Gaussian Dynamics for 4D Content Creation

**Authors**: Ulrich Neumann, Danhang Tang, Le Chen, Wenchao Ma, Ben Mildenhall, Zhe Cao, Qiangeng Xu, Quankai Gao

<details>
<summary><b>Abstract</b></summary>
Creating 4D fields of Gaussian Splatting from images or videos is a challenging task due to its under-constrained nature. While the optimization can draw photometric reference from the input videos or be regulated by generative models, directly supervising Gaussian motions remains underexplored. In this paper, we introduce a novel concept, Gaussian flow, which connects the dynamics of 3D Gaussians and pixel velocities between consecutive frames. The Gaussian flow can be efficiently obtained by splatting Gaussian dynamics into the image space. This differentiable process enables direct dynamic supervision from optical flow. Our method significantly benefits 4D dynamic content generation and 4D novel view synthesis with Gaussian Splatting, especially for contents with rich motions that are hard to be handled by existing methods. The common color drifting issue that happens in 4D generation is also resolved with improved Guassian dynamics. Superior visual quality on extensive experiments demonstrates our method's effectiveness. Quantitative and qualitative evaluations show that our method achieves state-of-the-art results on both tasks of 4D generation and 4D novel view synthesis. Project page: https://zerg-overmind.github.io/GaussianFlow.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12365v2.pdf)

#### MVDream: Multi-view Diffusion for 3D Generation

**Authors**: Xiao Yang, Kejie Li, Mai Long, Jianglong Ye, Peng Wang, Yichun Shi

<details>
<summary><b>Abstract</b></summary>
We introduce MVDream, a diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can achieve the generalizability of 2D diffusion models and the consistency of 3D renderings. We demonstrate that such a multi-view diffusion model is implicitly a generalizable 3D prior agnostic to 3D representations. It can be applied to 3D generation via Score Distillation Sampling, significantly enhancing the consistency and stability of existing 2D-lifting methods. It can also learn new concepts from a few 2D examples, akin to DreamBooth, but for 3D generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.16512v4.pdf)

#### HexPlane: A Fast Representation for Dynamic Scenes

**Authors**: Justin Johnson, Ang Cao

<details>
<summary><b>Abstract</b></summary>
Modeling and re-rendering dynamic 3D scenes is a challenging task in 3D vision. Prior approaches build on NeRF and rely on implicit representations. This is slow since it requires many MLP evaluations, constraining real-world applications. We show that dynamic 3D scenes can be explicitly represented by six planes of learned features, leading to an elegant solution we call HexPlane. A HexPlane computes features for points in spacetime by fusing vectors extracted from each plane, which is highly efficient. Pairing a HexPlane with a tiny MLP to regress output colors and training via volume rendering gives impressive results for novel view synthesis on dynamic scenes, matching the image quality of prior work but reducing training time by more than $100\times$. Extensive ablations confirm our HexPlane design and show that it is robust to different feature fusion mechanisms, coordinate systems, and decoding mechanisms. HexPlane is a simple and effective solution for representing 4D volumes, and we hope they can broadly contribute to modeling spacetime for dynamic 3D scenes.
</details>

[📄 Paper](https://arxiv.org/pdf/2301.09632v2.pdf)

#### BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors

**Authors**: Baoquan Chen, Libin Liu, Weiyu Li, Qingzhe Gao, Tingyang Zhang

<details>
<summary><b>Abstract</b></summary>
Animatable 3D reconstruction has significant applications across various fields, primarily relying on artists' handcraft creation. Recently, some studies have successfully constructed animatable 3D models from monocular videos. However, these approaches require sufficient view coverage of the object within the input video and typically necessitate significant time and computational costs for training and rendering. This limitation restricts the practical applications. In this work, we propose a method to build animatable 3D Gaussian Splatting from monocular video with diffusion priors. The 3D Gaussian representations significantly accelerate the training and rendering process, and the diffusion priors allow the method to learn 3D models with limited viewpoints. We also present the rigid regularization to enhance the utilization of the priors. We perform an extensive evaluation across various real-world videos, demonstrating its superior performance compared to the current state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11427v1.pdf)

#### 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency

**Authors**: Yunchao Wei, Yao Zhao, Zhangyang Wang, Dejia Xu, Yuyang Yin

<details>
<summary><b>Abstract</b></summary>
Aided by text-to-image and text-to-video diffusion models, existing 4D content creation pipelines utilize score distillation sampling to optimize the entire dynamic 3D scene. However, as these pipelines generate 4D content from text or image inputs, they incur significant time and effort in prompt engineering through trial and error. This work introduces 4DGen, a novel, holistic framework for grounded 4D content creation that decomposes the 4D generation task into multiple stages. We identify static 3D assets and monocular video sequences as key components in constructing the 4D content. Our pipeline facilitates conditional 4D generation, enabling users to specify geometry (3D assets) and motion (monocular videos), thus offering superior control over content creation. Furthermore, we construct our 4D representation using dynamic 3D Gaussians, which permits efficient, high-resolution supervision through rendering during training, thereby facilitating high-quality 4D generation. Additionally, we employ spatial-temporal pseudo labels on anchor frames, along with seamless consistency priors implemented through 3D-aware score distillation sampling and smoothness regularizations. Compared to existing baselines, our approach yields competitive results in faithfully reconstructing input signals and realistically inferring renderings from novel viewpoints and timesteps. Most importantly, our method supports grounded generation, offering users enhanced control, a feature difficult to achieve with previous methods. Project page: https://vita-group.github.io/4DGen/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.17225v2.pdf)

#### DreamGaussian4D: Generative 4D Gaussian Splatting

**Authors**: Ziwei Liu, Gang Zeng, Ang Cao, Chi Zhang, Jiaxiang Tang, Liang Pan, Jiawei Ren

<details>
<summary><b>Abstract</b></summary>
4D content generation has achieved remarkable progress recently. However, existing methods suffer from long optimization times, a lack of motion controllability, and a low quality of details. In this paper, we introduce DreamGaussian4D (DG4D), an efficient 4D generation framework that builds on Gaussian Splatting (GS). Our key insight is that combining explicit modeling of spatial transformations with static GS makes an efficient and powerful representation for 4D generation. Moreover, video generation methods have the potential to offer valuable spatial-temporal priors, enhancing the high-quality 4D generation. Specifically, we propose an integral framework with two major modules: 1) Image-to-4D GS - we initially generate static GS with DreamGaussianHD, followed by HexPlane-based dynamic generation with Gaussian deformation; and 2) Video-to-Video Texture Refinement - we refine the generated UV-space texture maps and meanwhile enhance their temporal consistency by utilizing a pre-trained image-to-video diffusion model. Notably, DG4D reduces the optimization time from several hours to just a few minutes, allows the generated 3D motion to be visually controlled, and produces animated meshes that can be realistically rendered in 3D engines.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.17142v3.pdf)

#### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

**Authors**: Gang Zeng, Ziwei Liu, Hang Zhou, Jiawei Ren, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.16653v2.pdf)

#### Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction

**Authors**: Houqiang Li, Min Wang, Li Li, Wengang Zhou, Zhiyang Guo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has become an emerging tool for dynamic scene reconstruction. However, existing methods focus mainly on extending static 3DGS into a time-variant representation, while overlooking the rich motion information carried by 2D observations, thus suffering from performance degradation and model redundancy. To address the above problem, we propose a novel motion-aware enhancement framework for dynamic scene reconstruction, which mines useful motion cues from optical flow to improve different paradigms of dynamic 3DGS. Specifically, we first establish a correspondence between 3D Gaussian movements and pixel-level flow. Then a novel flow augmentation method is introduced with additional insights into uncertainty and loss collaboration. Moreover, for the prevalent deformation-based paradigm that presents a harder optimization problem, a transient-aware deformation auxiliary module is proposed. We conduct extensive experiments on both multi-view and monocular scenes to verify the merits of our work. Compared with the baselines, our method shows significant superiority in both rendering quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11447v1.pdf)

#### Fast Dynamic 3D Object Generation from a Single-view Video

**Authors**: Li Zhang, Xiatian Zhu, Zeyu Yang, Zijie Pan

<details>
<summary><b>Abstract</b></summary>
Generating dynamic 3D object from a single-view video is challenging due to the lack of 4D labeled data. Extending image-to-3D pipelines by transferring off-the-shelf image generation models such as score distillation sampling, existing methods tend to be slow and expensive to scale due to the need for back-propagating the information-limited supervision signals through a large pretrained model. To address this, we propose an efficient video-to-4D object generation framework called Efficient4D. It generates high-quality spacetime-consistent images under different camera views, and then uses them as labeled data to directly train a novel 4D Gaussian splatting model with explicit point cloud geometry, enabling real-time rendering under continuous camera trajectories. Extensive experiments on synthetic and real videos show that Efficient4D offers a remarkable 20-fold increase in speed when compared to prior art alternatives while preserving the quality of novel view synthesis. For example, Efficient4D takes only 6 mins to model a dynamic object, vs 120 mins by Consistent4D.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.08742v2.pdf)

#### Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models

**Authors**: Karsten Kreis, Sanja Fidler, Antonio Torralba, Seung Wook Kim, Huan Ling

<details>
<summary><b>Abstract</b></summary>
Text-guided diffusion models have revolutionized image and video generation and have also been successfully used for optimization-based 3D object synthesis. Here, we instead focus on the underexplored text-to-4D setting and synthesize dynamic, animated 3D objects using score distillation methods with an additional temporal dimension. Compared to previous work, we pursue a novel compositional generation-based approach, and combine text-to-image, text-to-video, and 3D-aware multiview diffusion models to provide feedback during 4D object optimization, thereby simultaneously enforcing temporal consistency, high-quality visual appearance and realistic geometry. Our method, called Align Your Gaussians (AYG), leverages dynamic 3D Gaussian Splatting with deformation fields as 4D representation. Crucial to AYG is a novel method to regularize the distribution of the moving 3D Gaussians and thereby stabilize the optimization and induce motion. We also propose a motion amplification mechanism as well as a new autoregressive synthesis scheme to generate and combine multiple 4D sequences for longer generation. These techniques allow us to synthesize vivid dynamic scenes, outperform previous work qualitatively and quantitatively and achieve state-of-the-art text-to-4D performance. Due to the Gaussian 4D representation, different 4D animations can be seamlessly combined, as we demonstrate. AYG opens up promising avenues for animation, simulation and digital content creation as well as synthetic data generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13763v2.pdf)

#### Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

**Authors**: Li Zhang, Zijie Pan, Hongye Yang, Zeyu Yang

<details>
<summary><b>Abstract</b></summary>
Reconstructing dynamic 3D scenes from 2D images and generating diverse views over time is challenging due to scene complexity and temporal dynamics. Despite advancements in neural implicit models, limitations persist: (i) Inadequate Scene Structure: Existing methods struggle to reveal the spatial and temporal structure of dynamic scenes from directly learning the complex 6D plenoptic function. (ii) Scaling Deformation Modeling: Explicitly modeling scene element deformation becomes impractical for complex dynamics. To address these issues, we consider the spacetime as an entirety and propose to approximate the underlying spatio-temporal 4D volume of a dynamic scene by optimizing a collection of 4D primitives, with explicit geometry and appearance modeling. Learning to optimize the 4D primitives enables us to synthesize novel views at any desired time with our tailored rendering routine. Our model is conceptually simple, consisting of a 4D Gaussian parameterized by anisotropic ellipses that can rotate arbitrarily in space and time, as well as view-dependent and time-evolved appearance represented by the coefficient of 4D spherindrical harmonics. This approach offers simplicity, flexibility for variable-length video and end-to-end training, and efficient real-time rendering, making it suitable for capturing complex dynamic scene motions. Experiments across various benchmarks, including monocular and multi-view scenarios, demonstrate our 4DGS model's superior visual quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.10642v3.pdf)

## Autonomous Driving
### Autonomous Driving Scene Reconstruction
#### Street gaussians for modeling dynamic urban scenes
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting

**Authors**: Yiyi Liao, Andreas Geiger, Yue Wang, Bingbing Liu, Weichao Qiu, Dongfeng Bai, Lu Xu, Jiahao Shao, HongYu Zhou

<details>
<summary><b>Abstract</b></summary>
Holistic understanding of urban scenes based on RGB images is a challenging yet important problem. It encompasses understanding both the geometry and appearance to enable novel view synthesis, parsing semantic labels, and tracking moving objects. Despite considerable progress, existing approaches often focus on specific aspects of this task and require additional inputs such as LiDAR scans or manually annotated 3D bounding boxes. In this paper, we introduce a novel pipeline that utilizes 3D Gaussian Splatting for holistic urban scene understanding. Our main idea involves the joint optimization of geometry, appearance, semantics, and motion using a combination of static and dynamic 3D Gaussians, where moving object poses are regularized via physical constraints. Our approach offers the ability to render new viewpoints in real-time, yielding 2D and 3D semantic information with high accuracy, and reconstruct dynamic scenes, even in scenarios where 3D bounding box detection are highly noisy. Experimental results on KITTI, KITTI-360, and Virtual KITTI 2 demonstrate the effectiveness of our approach.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12722v1.pdf)

#### 3DGS-Calib: 3D Gaussian Splatting for Multimodal SpatioTemporal Calibration

**Authors**: Cédric Demonceaux, Pascal Vasseur, Cyrille Migniot, Dzmitry Tsishkou, Luis Roldao, Nathan Piasco, Arthur Moreau, Moussab Bennehar, Quentin Herau

<details>
<summary><b>Abstract</b></summary>
Reliable multimodal sensor fusion algorithms re- quire accurate spatiotemporal calibration. Recently, targetless calibration techniques based on implicit neural representations have proven to provide precise and robust results. Nevertheless, such methods are inherently slow to train given the high compu- tational overhead caused by the large number of sampled points required for volume rendering. With the recent introduction of 3D Gaussian Splatting as a faster alternative to implicit representation methods, we propose to leverage this new ren- dering approach to achieve faster multi-sensor calibration. We introduce 3DGS-Calib, a new calibration method that relies on the speed and rendering accuracy of 3D Gaussian Splatting to achieve multimodal spatiotemporal calibration that is accurate, robust, and with a substantial speed-up compared to methods relying on implicit neural representations. We demonstrate the superiority of our proposal with experimental results on sequences from KITTI-360, a widely used driving dataset.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11577v1.pdf)

#### DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes

**Authors**: Ming-Hsuan Yang, Deqing Sun, Yongtao Wang, Xiaojun Shan, Zhiwei Lin, Xiaoyu Zhou

<details>
<summary><b>Abstract</b></summary>
We present DrivingGaussian, an efficient and effective framework for surrounding dynamic autonomous driving scenes. For complex scenes with moving objects, we first sequentially and progressively model the static background of the entire scene with incremental static 3D Gaussians. We then leverage a composite dynamic Gaussian graph to handle multiple moving objects, individually reconstructing each object and restoring their accurate positions and occlusion relationships within the scene. We further use a LiDAR prior for Gaussian Splatting to reconstruct scenes with greater details and maintain panoramic consistency. DrivingGaussian outperforms existing methods in dynamic driving scene reconstruction and enables photorealistic surround-view synthesis with high-fidelity and multi-camera consistency. Our project page is at: https://github.com/VDIGPKU/DrivingGaussian.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.07920v3.pdf)

### Simultaneous Localization and Mapping (SLAM)
#### High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization

**Authors**: Martin Magnusson, Achim J. Lilienthal, Malcolm Mielle, Shuo Sun

<details>
<summary><b>Abstract</b></summary>
We propose a dense RGBD SLAM system based on 3D Gaussian Splatting that provides metrically accurate pose tracking and visually realistic reconstruction. To this end, we first propose a Gaussian densification strategy based on the rendering loss to map unobserved areas and refine reobserved areas. Second, we introduce extra regularization parameters to alleviate the forgetting problem in the continuous mapping problem, where parameters tend to overfit the latest frame and result in decreasing rendering quality for previous frames. Both mapping and tracking are performed with Gaussian parameters by minimizing re-rendering loss in a differentiable way. Compared to recent neural and concurrently developed gaussian splatting RGBD SLAM baselines, our method achieves state-of-the-art results on the synthetic dataset Replica and competitive results on the real-world dataset TUM.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.12535v1.pdf)

#### GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

**Authors**: Dong Wang, Bin Zhao, Xuelong Li, Zhigang Wang, Dan Xu, Delin Qu, Chi Yan

<details>
<summary><b>Abstract</b></summary>
In this paper, we introduce \textbf{GS-SLAM} that first utilizes 3D Gaussian representation in the Simultaneous Localization and Mapping (SLAM) system. It facilitates a better balance between efficiency and accuracy. Compared to recent SLAM methods employing neural implicit representations, our method utilizes a real-time differentiable splatting rendering pipeline that offers significant speedup to map optimization and RGB-D rendering. Specifically, we propose an adaptive expansion strategy that adds new or deletes noisy 3D Gaussians in order to efficiently reconstruct new observed scene geometry and improve the mapping of previously observed areas. This strategy is essential to extend 3D Gaussian representation to reconstruct the whole scene rather than synthesize a static object in existing methods. Moreover, in the pose tracking process, an effective coarse-to-fine technique is designed to select reliable 3D Gaussian representations to optimize camera pose, resulting in runtime reduction and robust estimation. Our method achieves competitive performance compared with existing state-of-the-art real-time methods on the Replica, TUM-RGBD datasets. Project page: https://gs-slam.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11700v4.pdf)

#### Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras

**Authors**: Sai-Kit Yeung, Hui Cheng, Longwei Li, Huajian Huang

<details>
<summary><b>Abstract</b></summary>
The integration of neural rendering and the SLAM system recently showed promising results in joint localization and photorealistic view reconstruction. However, existing methods, fully relying on implicit representations, are so resource-hungry that they cannot run on portable devices, which deviates from the original intention of SLAM. In this paper, we present Photo-SLAM, a novel SLAM framework with a hyper primitives map. Specifically, we simultaneously exploit explicit geometric features for localization and learn implicit photometric features to represent the texture information of the observed environment. In addition to actively densifying hyper primitives based on geometric features, we further introduce a Gaussian-Pyramid-based training method to progressively learn multi-level features, enhancing photorealistic mapping performance. The extensive experiments with monocular, stereo, and RGB-D datasets prove that our proposed system Photo-SLAM significantly outperforms current state-of-the-art SLAM systems for online photorealistic mapping, e.g., PSNR is 30% higher and rendering speed is hundreds of times faster in the Replica dataset. Moreover, the Photo-SLAM can run at real-time speed using an embedded platform such as Jetson AGX Orin, showing the potential of robotics applications.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16728v2.pdf)

#### SGS-SLAM: Semantic Gaussian Splatting For Neural Dense SLAM

**Authors**: Tianchen Deng, Hongyu Wang, Na Cheng, Guohao Zhu, Heng Zhou, Shuhong Liu, Mingrui Li

<details>
<summary><b>Abstract</b></summary>
We present SGS-SLAM, the first semantic visual SLAM system based on Gaussian Splatting. It incorporates appearance, geometry, and semantic features through multi-channel optimization, addressing the oversmoothing limitations of neural implicit SLAM systems in high-quality rendering, scene understanding, and object-level geometry. We introduce a unique semantic feature loss that effectively compensates for the shortcomings of traditional depth and color losses in object optimization. Through a semantic-guided keyframe selection strategy, we prevent erroneous reconstructions caused by cumulative errors. Extensive experiments demonstrate that SGS-SLAM delivers state-of-the-art performance in camera pose estimation, map reconstruction, precise semantic segmentation, and object-level geometric accuracy, while ensuring real-time rendering capabilities.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.03246v5.pdf)

#### NEDS-SLAM: A Novel Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting

**Authors**: Zongwu Xie, Boyu Ma, Guanghu Xie, Yang Liu, Yiming Ji

<details>
<summary><b>Abstract</b></summary>
We propose NEDS-SLAM, an Explicit Dense semantic SLAM system based on 3D Gaussian representation, that enables robust 3D semantic mapping, accurate camera tracking, and high-quality rendering in real-time. In the system, we propose a Spatially Consistent Feature Fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic Gaussian mapping. Additionally, we employ a lightweight encoder-decoder to compress the high-dimensional semantic features into a compact 3D Gaussian representation, mitigating the burden of excessive memory consumption. Furthermore, we leverage the advantage of 3D Gaussian splatting, which enables efficient and differentiable novel view rendering, and propose a Virtual Camera View Pruning method to eliminate outlier GS points, thereby effectively enhancing the quality of scene representations. Our NEDS-SLAM method demonstrates competitive performance over existing dense semantic SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in 3D dense semantic mapping.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11679v2.pdf)

#### Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data

**Authors**: Hengshuang Zhao, Jiashi Feng, Xiaogang Xu, Zilong Huang, Bingyi Kang, Lihe Yang

<details>
<summary><b>Abstract</b></summary>
This work presents Depth Anything, a highly practical solution for robust monocular depth estimation. Without pursuing novel technical modules, we aim to build a simple yet powerful foundation model dealing with any images under any circumstances. To this end, we scale up the dataset by designing a data engine to collect and automatically annotate large-scale unlabeled data (~62M), which significantly enlarges the data coverage and thus is able to reduce the generalization error. We investigate two simple yet effective strategies that make data scaling-up promising. First, a more challenging optimization target is created by leveraging data augmentation tools. It compels the model to actively seek extra visual knowledge and acquire robust representations. Second, an auxiliary supervision is developed to enforce the model to inherit rich semantic priors from pre-trained encoders. We evaluate its zero-shot capabilities extensively, including six public datasets and randomly captured photos. It demonstrates impressive generalization ability. Further, through fine-tuning it with metric depth information from NYUv2 and KITTI, new SOTAs are set. Our better depth model also results in a better depth-conditioned ControlNet. Our models are released at https://github.com/LiheYoung/Depth-Anything.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.10891v2.pdf)

#### DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras

**Authors**: Jia Deng, Zachary Teed

<details>
<summary><b>Abstract</b></summary>
We introduce DROID-SLAM, a new deep learning based SLAM system. DROID-SLAM consists of recurrent iterative updates of camera pose and pixelwise depth through a Dense Bundle Adjustment layer. DROID-SLAM is accurate, achieving large improvements over prior work, and robust, suffering from substantially fewer catastrophic failures. Despite training on monocular video, it can leverage stereo or RGB-D video to achieve improved performance at test time. The URL to our open source code is https://github.com/princeton-vl/DROID-SLAM.
</details>

[📄 Paper](https://arxiv.org/pdf/2108.10869v2.pdf)

#### SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM

**Authors**: Hesheng Wang, Jiuming Liu, Guangming Wang, Renjie Qin, Siting Zhu

<details>
<summary><b>Abstract</b></summary>
We propose SemGauss-SLAM, a dense semantic SLAM system utilizing 3D Gaussian representation, that enables accurate 3D semantic mapping, robust camera tracking, and high-quality rendering simultaneously. In this system, we incorporate semantic feature embedding into 3D Gaussian representation, which effectively encodes semantic information within the spatial layout of the environment for precise semantic scene representation. Furthermore, we propose feature-level loss for updating 3D Gaussian representation, enabling higher-level guidance for 3D Gaussian optimization. In addition, to reduce cumulative drift in tracking and improve semantic reconstruction accuracy, we introduce semantic-informed bundle adjustment leveraging multi-frame semantic associations for joint optimization of 3D Gaussian representation and camera poses, leading to low-drift tracking and accurate mapping. Our SemGauss-SLAM method demonstrates superior performance over existing radiance field-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in high-precision semantic segmentation and dense semantic mapping.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.07494v3.pdf)

#### Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting

**Authors**: Martin R. Oswald, Theo Gevers, Yue Li, Vladimir Yugay

<details>
<summary><b>Abstract</b></summary>
We present a dense simultaneous localization and mapping (SLAM) method that uses 3D Gaussians as a scene representation. Our approach enables interactive-time reconstruction and photo-realistic rendering from real-world single-camera RGBD videos. To this end, we propose a novel effective strategy for seeding new Gaussians for newly explored areas and their effective online optimization that is independent of the scene size and thus scalable to larger scenes. This is achieved by organizing the scene into sub-maps which are independently optimized and do not need to be kept in memory. We further accomplish frame-to-model camera tracking by minimizing photometric and geometric losses between the input and rendered frames. The Gaussian representation allows for high-quality photo-realistic real-time rendering of real-world scenes. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance in mapping, tracking, and rendering compared to existing neural dense SLAM methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.10070v2.pdf)

#### GaussNav: Gaussian Splatting for Visual Navigation

**Authors**: Houqiang Li, Wengang Zhou, Min Wang, Xiaohan Lei

<details>
<summary><b>Abstract</b></summary>
In embodied vision, Instance ImageGoal Navigation (IIN) requires an agent to locate a specific object depicted in a goal image within an unexplored environment. The primary difficulty of IIN stems from the necessity of recognizing the target object across varying viewpoints and rejecting potential distractors. Existing map-based navigation methods largely adopt the representation form of Bird's Eye View (BEV) maps, which, however, lack the representation of detailed textures in a scene. To address the above issues, we propose a new Gaussian Splatting Navigation (abbreviated as GaussNav) framework for IIN task, which constructs a novel map representation based on 3D Gaussian Splatting (3DGS). The proposed framework enables the agent to not only memorize the geometry and semantic information of the scene, but also retain the textural features of objects. Our GaussNav framework demonstrates a significant leap in performance, evidenced by an increase in Success weighted by Path Length (SPL) from 0.252 to 0.578 on the challenging Habitat-Matterport 3D (HM3D) dataset. Our code will be made publicly available.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11625v2.pdf)

#### RGBD GS-ICP SLAM

**Authors**: Hyeonwoo Yu, Jiung Yeon, Seongbo Ha

<details>
<summary><b>Abstract</b></summary>
Simultaneous Localization and Mapping (SLAM) with dense representation plays a key role in robotics, Virtual Reality (VR), and Augmented Reality (AR) applications. Recent advancements in dense representation SLAM have highlighted the potential of leveraging neural scene representation and 3D Gaussian representation for high-fidelity spatial representation. In this paper, we propose a novel dense representation SLAM approach with a fusion of Generalized Iterative Closest Point (G-ICP) and 3D Gaussian Splatting (3DGS). In contrast to existing methods, we utilize a single Gaussian map for both tracking and mapping, resulting in mutual benefits. Through the exchange of covariances between tracking and mapping processes with scale alignment techniques, we minimize redundant computations and achieve an efficient system. Additionally, we enhance tracking accuracy and mapping quality through our keyframe selection methods. Experimental results demonstrate the effectiveness of our approach, showing an incredibly fast speed up to 107 FPS (for the entire system) and superior quality of the reconstructed map.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12550v2.pdf)

#### Joint Scene and Object Tracking for Cost-Effective Augmented Reality Assisted Patient Positioning in Radiation Therapy

**Authors**: R. Medina-Carnicer, Antonio Luna, M. Álvaro Berbís, Rafael Muñoz-Salinas, Hamid Sarmadi

<details>
<summary><b>Abstract</b></summary>
BACKGROUND AND OBJECTIVE: The research done in the field of Augmented Reality (AR) for patient positioning in radiation therapy is scarce. We propose an efficient and cost-effective algorithm for tracking the scene and the patient to interactively assist the patient's positioning process by providing visual feedback to the operator. Up to our knowledge, this is the first framework that can be employed for mobile interactive AR to guide patient positioning. METHODS: We propose a point cloud processing method that combined with a fiducial marker-mapper algorithm and the generalized ICP algorithm tracks the patient and the camera precisely and efficiently only using the CPU unit. The alignment between the 3D reference model and body marker map is calculated employing an efficient body reconstruction algorithm. RESULTS: Our quantitative evaluation shows that the proposed method achieves a translational and rotational error of 4.17 mm/0.82 deg at 9 fps. Furthermore, the qualitative results demonstrate the usefulness of our algorithm in patient positioning on different human subjects. CONCLUSION: Since our algorithm achieves a relatively high frame rate and accuracy employing a regular laptop (without the usage of a dedicated GPU), it is a very cost-effective AR-based patient positioning method. It also opens the way for other researchers by introducing a framework that could be improved upon for better mobile interactive AR patient positioning solutions in the future.
</details>

[📄 Paper](https://arxiv.org/pdf/2010.01895v2.pdf)

#### 3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization

**Authors**: Srikanth Saripalli, Gaurav Pandey, Peng Jiang

<details>
<summary><b>Abstract</b></summary>
This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11367v1.pdf)

#### How NeRFs and 3D Gaussian Splatting are Reshaping SLAM: a Survey

**Authors**: Matteo Poggi, Martin R. Oswald, Stefano Mattoccia, Erik Sandström, Ziren Gong, Youmin Zhang, Fabio Tosi

<details>
<summary><b>Abstract</b></summary>
Over the past two decades, research in the field of Simultaneous Localization and Mapping (SLAM) has undergone a significant evolution, highlighting its critical role in enabling autonomous exploration of unknown environments. This evolution ranges from hand-crafted methods, through the era of deep learning, to more recent developments focused on Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) representations. Recognizing the growing body of research and the absence of a comprehensive survey on the topic, this paper aims to provide the first comprehensive overview of SLAM progress through the lens of the latest advancements in radiance fields. It sheds light on the background, evolutionary path, inherent strengths and limitations, and serves as a fundamental reference to highlight the dynamic progress and specific challenges.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.13255v2.pdf)

#### Gaussian Splatting SLAM

**Authors**: Andrew J. Davison, Paul H. J. Kelly, Riku Murai, Hidenobu Matsuki

<details>
<summary><b>Abstract</b></summary>
We present the first application of 3D Gaussian Splatting in monocular SLAM, the most fundamental but the hardest setup for Visual SLAM. Our method, which runs live at 3fps, utilises Gaussians as the only 3D representation, unifying the required representation for accurate, efficient tracking, mapping, and high-quality rendering. Designed for challenging monocular settings, our approach is seamlessly extendable to RGB-D SLAM when an external depth sensor is available. Several innovations are required to continuously reconstruct 3D scenes with high fidelity from a live camera. First, to move beyond the original 3DGS algorithm, which requires accurate poses from an offline Structure from Motion (SfM) system, we formulate camera tracking for 3DGS using direct optimisation against the 3D Gaussians, and show that this enables fast and robust tracking with a wide basin of convergence. Second, by utilising the explicit nature of the Gaussians, we introduce geometric verification and regularisation to handle the ambiguities occurring in incremental 3D dense reconstruction. Finally, we introduce a full SLAM system which not only achieves state-of-the-art results in novel view synthesis and trajectory estimation but also reconstruction of tiny and even transparent objects.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.06741v2.pdf)

#### SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM

**Authors**: Jonathon Luiten, Deva Ramanan, Sebastian Scherer, Gengshan Yang, Krishna Murthy Jatavallabhula, Jay Karhade, Nikhil Keetha

<details>
<summary><b>Abstract</b></summary>
Dense simultaneous localization and mapping (SLAM) is crucial for robotics and augmented reality applications. However, current methods are often hampered by the non-volumetric or implicit way they represent a scene. This work introduces SplaTAM, an approach that, for the first time, leverages explicit volumetric representations, i.e., 3D Gaussians, to enable high-fidelity reconstruction from a single unposed RGB-D camera, surpassing the capabilities of existing methods. SplaTAM employs a simple online tracking and mapping system tailored to the underlying Gaussian representation. It utilizes a silhouette mask to elegantly capture the presence of scene density. This combination enables several benefits over prior representations, including fast rendering and dense optimization, quickly determining if areas have been previously mapped, and structured map expansion by adding more Gaussians. Extensive experiments show that SplaTAM achieves up to 2x superior performance in camera pose estimation, map construction, and novel-view synthesis over existing methods, paving the way for more immersive high-fidelity SLAM applications.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02126v3.pdf)

# Extensions of 3D Gaussian Splatting
## Dynamic 3D Gaussian Splatting
### Multi-view Videos
#### SWAGS: Sampling Windows Adaptively for Dynamic 3D Gaussian Splatting
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### Instant Neural Graphics Primitives with a Multiresolution Hash Encoding

**Authors**: Alexander Keller, Christoph Schied, Alex Evans, Thomas Müller

<details>
<summary><b>Abstract</b></summary>
Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a smaller network without sacrificing quality, thus significantly reducing the number of floating point and memory access operations: a small neural network is augmented by a multiresolution hash table of trainable feature vectors whose values are optimized through stochastic gradient descent. The multiresolution structure allows the network to disambiguate hash collisions, making for a simple architecture that is trivial to parallelize on modern GPUs. We leverage this parallelism by implementing the whole system using fully-fused CUDA kernels with a focus on minimizing wasted bandwidth and compute operations. We achieve a combined speedup of several orders of magnitude, enabling training of high-quality neural graphics primitives in a matter of seconds, and rendering in tens of milliseconds at a resolution of ${1920\!\times\!1080}$.
</details>

[📄 Paper](https://arxiv.org/pdf/2201.05989v2.pdf)

#### Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

**Authors**: Deva Ramanan, Bastian Leibe, Georgios Kopanas, Jonathon Luiten

<details>
<summary><b>Abstract</b></summary>
We present a method that simultaneously addresses the tasks of dynamic scene novel-view synthesis and six degree-of-freedom (6-DOF) tracking of all dense scene elements. We follow an analysis-by-synthesis framework, inspired by recent work that models scenes as a collection of 3D Gaussians which are optimized to reconstruct input images via differentiable rendering. To model dynamic scenes, we allow Gaussians to move and rotate over time while enforcing that they have persistent color, opacity, and size. By regularizing Gaussians' motion and rotation with local-rigidity constraints, we show that our Dynamic 3D Gaussians correctly model the same area of physical space over time, including the rotation of that space. Dense 6-DOF tracking and dynamic reconstruction emerges naturally from persistent dynamic view synthesis, without requiring any correspondence or flow as input. We demonstrate a large number of downstream applications enabled by our representation, including first-person view synthesis, dynamic compositional scene synthesis, and 4D video editing.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.09713v1.pdf)

#### 3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos

**Authors**: Wei Xing, Lei Zhao, Zhanjie Zhang, Guangyuan Li, Han Jiao, Jiakai Sun

<details>
<summary><b>Abstract</b></summary>
Constructing photo-realistic Free-Viewpoint Videos (FVVs) of dynamic scenes from multi-view videos remains a challenging endeavor. Despite the remarkable advancements achieved by current neural rendering techniques, these methods generally require complete video sequences for offline training and are not capable of real-time rendering. To address these constraints, we introduce 3DGStream, a method designed for efficient FVV streaming of real-world dynamic scenes. Our method achieves fast on-the-fly per-frame reconstruction within 12 seconds and real-time rendering at 200 FPS. Specifically, we utilize 3D Gaussians (3DGs) to represent the scene. Instead of the na\"ive approach of directly optimizing 3DGs per-frame, we employ a compact Neural Transformation Cache (NTC) to model the translations and rotations of 3DGs, markedly reducing the training time and storage required for each FVV frame. Furthermore, we propose an adaptive 3DG addition strategy to handle emerging objects in dynamic scenes. Experiments demonstrate that 3DGStream achieves competitive performance in terms of rendering speed, image quality, training time, and model storage when compared with state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.01444v4.pdf)

#### Bridging 3D Gaussian and Mesh for Freeview Video Rendering

**Authors**: Shenghua Gao, Yujun Shen, Minghui Yang, Nan Xue, Yanbo Fan, Hongrui Cai, Jiafei Li, Xuan Wang, Yuting Xiao

<details>
<summary><b>Abstract</b></summary>
This is only a preview version of GauMesh. Recently, primitive-based rendering has been proven to achieve convincing results in solving the problem of modeling and rendering the 3D dynamic scene from 2D images. Despite this, in the context of novel view synthesis, each type of primitive has its inherent defects in terms of representation ability. It is difficult to exploit the mesh to depict the fuzzy geometry. Meanwhile, the point-based splatting (e.g. the 3D Gaussian Splatting) method usually produces artifacts or blurry pixels in the area with smooth geometry and sharp textures. As a result, it is difficult, even not impossible, to represent the complex and dynamic scene with a single type of primitive. To this end, we propose a novel approach, GauMesh, to bridge the 3D Gaussian and Mesh for modeling and rendering the dynamic scenes. Given a sequence of tracked mesh as initialization, our goal is to simultaneously optimize the mesh geometry, color texture, opacity maps, a set of 3D Gaussians, and the deformation field. At a specific time, we perform $\alpha$-blending on the RGB and opacity values based on the merged and re-ordered z-buffers from mesh and 3D Gaussian rasterizations. This produces the final rendering, which is supervised by the ground-truth image. Experiments demonstrate that our approach adapts the appropriate type of primitives to represent the different parts of the dynamic scene and outperforms all the baseline methods in both quantitative and qualitative comparisons without losing render speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11453v1.pdf)

### Monocular Video
#### Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction

**Authors**: Xiaogang Jin, Yuqing Zhang, Shaohui Jiao, Wen Zhou, Xinyu Gao, ZiYi Yang

<details>
<summary><b>Abstract</b></summary>
Implicit neural representation has paved the way for new approaches to dynamic scene reconstruction and rendering. Nonetheless, cutting-edge dynamic neural rendering methods rely heavily on these implicit representations, which frequently struggle to capture the intricate details of objects in the scene. Furthermore, implicit methods have difficulty achieving real-time rendering in general dynamic scenes, limiting their use in a variety of tasks. To address the issues, we propose a deformable 3D Gaussians Splatting method that reconstructs scenes using 3D Gaussians and learns them in canonical space with a deformation field to model monocular dynamic scenes. We also introduce an annealing smoothing training mechanism with no extra overhead, which can mitigate the impact of inaccurate poses on the smoothness of time interpolation tasks in real-world datasets. Through a differential Gaussian rasterizer, the deformable 3D Gaussians not only achieve higher rendering quality but also real-time rendering speed. Experiments show that our method outperforms existing methods significantly in terms of both rendering quality and speed, making it well-suited for tasks such as novel-view synthesis, time interpolation, and real-time rendering.
</details>

[📄 Paper](https://arxiv.org/pdf/2309.13101v2.pdf)

#### 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

**Authors**: Li Zhang, Zijie Pan, Hongye Yang, Zeyu Yang

<details>
<summary><b>Abstract</b></summary>
Reconstructing dynamic 3D scenes from 2D images and generating diverse views over time is challenging due to scene complexity and temporal dynamics. Despite advancements in neural implicit models, limitations persist: (i) Inadequate Scene Structure: Existing methods struggle to reveal the spatial and temporal structure of dynamic scenes from directly learning the complex 6D plenoptic function. (ii) Scaling Deformation Modeling: Explicitly modeling scene element deformation becomes impractical for complex dynamics. To address these issues, we consider the spacetime as an entirety and propose to approximate the underlying spatio-temporal 4D volume of a dynamic scene by optimizing a collection of 4D primitives, with explicit geometry and appearance modeling. Learning to optimize the 4D primitives enables us to synthesize novel views at any desired time with our tailored rendering routine. Our model is conceptually simple, consisting of a 4D Gaussian parameterized by anisotropic ellipses that can rotate arbitrarily in space and time, as well as view-dependent and time-evolved appearance represented by the coefficient of 4D spherindrical harmonics. This approach offers simplicity, flexibility for variable-length video and end-to-end training, and efficient real-time rendering, making it suitable for capturing complex dynamic scene motions. Experiments across various benchmarks, including monocular and multi-view scenarios, demonstrate our 4DGS model's superior visual quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.10642v3.pdf)

#### GauFRe: Gaussian Deformation Fields for Real-time Dynamic Novel View Synthesis

**Authors**: Lei Xiao, James Tompkin, Douglas Lanman, Thu Nguyen-Phuoc, Zhengqin Li, Numair Khan, Yiqing Liang

<details>
<summary><b>Abstract</b></summary>
We propose a method for dynamic scene reconstruction using deformable 3D Gaussians that is tailored for monocular video. Building upon the efficiency of Gaussian splatting, our approach extends the representation to accommodate dynamic elements via a deformable set of Gaussians residing in a canonical space, and a time-dependent deformation field defined by a multi-layer perceptron (MLP). Moreover, under the assumption that most natural scenes have large regions that remain static, we allow the MLP to focus its representational power by additionally including a static Gaussian point cloud. The concatenated dynamic and static point clouds form the input for the Gaussian Splatting rasterizer, enabling real-time rendering. The differentiable pipeline is optimized end-to-end with a self-supervised rendering loss. Our method achieves results that are comparable to state-of-the-art dynamic neural radiance field methods while allowing much faster optimization and rendering. Project website: https://lynl7130.github.io/gaufre/index.html
</details>
[📄 Paper](https://arxiv.org/pdf/2312.11458v1.pdf)

#### SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

**Authors**: Xiaojuan Qi, Yan-Pei Cao, Xiaoyang Lyu, ZiYi Yang, Yang-tian Sun, Yi-Hua Huang

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis for dynamic scenes is still a challenging problem in computer vision and graphics. Recently, Gaussian splatting has emerged as a robust technique to represent static scenes and enable high-quality and real-time novel view synthesis. Building upon this technique, we propose a new representation that explicitly decomposes the motion and appearance of dynamic scenes into sparse control points and dense Gaussians, respectively. Our key idea is to use sparse control points, significantly fewer in number than the Gaussians, to learn compact 6 DoF transformation bases, which can be locally interpolated through learned interpolation weights to yield the motion field of 3D Gaussians. We employ a deformation MLP to predict time-varying 6 DoF transformations for each control point, which reduces learning complexities, enhances learning abilities, and facilitates obtaining temporal and spatial coherent motion patterns. Then, we jointly learn the 3D Gaussians, the canonical space locations of control points, and the deformation MLP to reconstruct the appearance, geometry, and dynamics of 3D scenes. During learning, the location and number of control points are adaptively adjusted to accommodate varying motion complexities in different regions, and an ARAP loss following the principle of as rigid as possible is developed to enforce spatial continuity and local rigidity of learned motions. Finally, thanks to the explicit sparse motion representation and its decomposition from appearance, our method can enable user-controlled motion editing while retaining high-fidelity appearances. Extensive experiments demonstrate that our approach outperforms existing approaches on novel view synthesis with a high rendering speed and enables novel appearance-preserved motion editing applications. Project page: https://yihua7.github.io/SC-GS-web/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.14937v3.pdf)

#### Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

**Authors**: Deva Ramanan, Bastian Leibe, Georgios Kopanas, Jonathon Luiten

<details>
<summary><b>Abstract</b></summary>
We present a method that simultaneously addresses the tasks of dynamic scene novel-view synthesis and six degree-of-freedom (6-DOF) tracking of all dense scene elements. We follow an analysis-by-synthesis framework, inspired by recent work that models scenes as a collection of 3D Gaussians which are optimized to reconstruct input images via differentiable rendering. To model dynamic scenes, we allow Gaussians to move and rotate over time while enforcing that they have persistent color, opacity, and size. By regularizing Gaussians' motion and rotation with local-rigidity constraints, we show that our Dynamic 3D Gaussians correctly model the same area of physical space over time, including the rotation of that space. Dense 6-DOF tracking and dynamic reconstruction emerges naturally from persistent dynamic view synthesis, without requiring any correspondence or flow as input. We demonstrate a large number of downstream applications enabled by our representation, including first-person view synthesis, dynamic compositional scene synthesis, and 4D video editing.
</details>
[📄 Paper](https://arxiv.org/pdf/2308.09713v1.pdf)

#### DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting

**Authors**: Kostas Daniilidis, Jiahui Lei, Agelos Kratimenos

<details>
<summary><b>Abstract</b></summary>
Accurately and efficiently modeling dynamic scenes and motions is considered so challenging a task due to temporal dynamics and motion complexity. To address these challenges, we propose DynMF, a compact and efficient representation that decomposes a dynamic scene into a few neural trajectories. We argue that the per-point motions of a dynamic scene can be decomposed into a small set of explicit or learned trajectories. Our carefully designed neural framework consisting of a tiny set of learned basis queried only in time allows for rendering speed similar to 3D Gaussian Splatting, surpassing 120 FPS, while at the same time, requiring only double the storage compared to static scenes. Our neural representation adequately constrains the inherently underconstrained motion field of a dynamic scene leading to effective and fast optimization. This is done by biding each point to motion coefficients that enforce the per-point sharing of basis trajectories. By carefully applying a sparsity loss to the motion coefficients, we are able to disentangle the motions that comprise the scene, independently control them, and generate novel motion combinations that have never been seen before. We can reach state-of-the-art render quality within just 5 minutes of training and in less than half an hour, we can synthesize novel views of dynamic scenes with superior photorealistic quality. Our representation is interpretable, efficient, and expressive enough to offer real-time view synthesis of complex dynamic scene motions, in monocular and multi-view scenarios.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00112v1.pdf)

#### MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes

**Authors**: Jeffrey Ichnowski, Shuran Song, Mike Zheng Shou, Jia-Wei Liu, Yunchao Yao, Zhao Mandi, Bardienus P. Duisterhof

<details>
<summary><b>Abstract</b></summary>
Accurate 3D tracking in highly deformable scenes with occlusions and shadows can facilitate new applications in robotics, augmented reality, and generative AI. However, tracking under these conditions is extremely challenging due to the ambiguity that arises with large deformations, shadows, and occlusions. We introduce MD-Splatting, an approach for simultaneous 3D tracking and novel view synthesis, using video captures of a dynamic scene from various camera poses. MD-Splatting builds on recent advances in Gaussian splatting, a method that learns the properties of a large number of Gaussians for state-of-the-art and fast novel view synthesis. MD-Splatting learns a deformation function to project a set of Gaussians with non-metric, thus canonical, properties into metric space. The deformation function uses a neural-voxel encoding and a multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow scalar. We enforce physics-inspired regularization terms based on local rigidity, conservation of momentum, and isometry, which leads to trajectories with smaller trajectory errors. MD-Splatting achieves high-quality 3D tracking on highly deformable scenes with shadows and occlusions. Compared to state-of-the-art, we improve 3D tracking by an average of 23.9 %, while simultaneously achieving high-quality novel view synthesis. With sufficient texture such as in scene 6, MD-Splatting achieves a median tracking error of 3.39 mm on a cloth of 1 x 1 meters in size. Project website: https://md-splatting.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00583v1.pdf)

#### Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction

**Authors**: Houqiang Li, Min Wang, Li Li, Wengang Zhou, Zhiyang Guo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has become an emerging tool for dynamic scene reconstruction. However, existing methods focus mainly on extending static 3DGS into a time-variant representation, while overlooking the rich motion information carried by 2D observations, thus suffering from performance degradation and model redundancy. To address the above problem, we propose a novel motion-aware enhancement framework for dynamic scene reconstruction, which mines useful motion cues from optical flow to improve different paradigms of dynamic 3DGS. Specifically, we first establish a correspondence between 3D Gaussian movements and pixel-level flow. Then a novel flow augmentation method is introduced with additional insights into uncertainty and loss collaboration. Moreover, for the prevalent deformation-based paradigm that presents a harder optimization problem, a transient-aware deformation auxiliary module is proposed. We conduct extensive experiments on both multi-view and monocular scenes to verify the merits of our work. Compared with the baselines, our method shows significant superiority in both rendering quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11447v1.pdf)

#### An efficient 3d gaussian representation for monocular/multi-view dynamic scenes
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle

**Authors**: Yao Yao, Siyu Zhu, Zuozhuo Dai, Youtian Lin

<details>
<summary><b>Abstract</b></summary>
We introduce Gaussian-Flow, a novel point-based approach for fast dynamic scene reconstruction and real-time rendering from both multi-view and monocular videos. In contrast to the prevalent NeRF-based approaches hampered by slow training and rendering speeds, our approach harnesses recent advancements in point-based 3D Gaussian Splatting (3DGS). Specifically, a novel Dual-Domain Deformation Model (DDDM) is proposed to explicitly model attribute deformations of each Gaussian point, where the time-dependent residual of each attribute is captured by a polynomial fitting in the time domain, and a Fourier series fitting in the frequency domain. The proposed DDDM is capable of modeling complex scene deformations across long video footage, eliminating the need for training separate 3DGS for each frame or introducing an additional implicit neural field to model 3D dynamics. Moreover, the explicit deformation modeling for discretized Gaussian points ensures ultra-fast training and rendering of a 4D scene, which is comparable to the original 3DGS designed for static 3D reconstruction. Our proposed approach showcases a substantial efficiency improvement, achieving a $5\times$ faster training speed compared to the per-frame 3DGS modeling. In addition, quantitative results demonstrate that the proposed Gaussian-Flow significantly outperforms previous leading methods in novel view rendering quality. Project page: https://nju-3dv.github.io/projects/Gaussian-Flow
</details>

[📄 Paper](https://arxiv.org/pdf/2312.03431v1.pdf)

#### 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

**Authors**: Xinggang Wang, Qi Tian, Wenyu Liu, Wei Wei, Xiaopeng Zhang, Lingxi Xie, Jiemin Fang, Taoran Yi, Guanjun Wu

<details>
<summary><b>Abstract</b></summary>
Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800$\times$800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods. More demos and code are available at https://guanjunwu.github.io/4dgs/.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.08528v3.pdf)

#### Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

**Authors**: Yi Xu, Zhong Li, Zhang Chen, Zhan Li

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis of dynamic scenes has been an intriguing yet challenging problem. Despite recent advancements, simultaneously achieving high-resolution photorealistic results, real-time rendering, and compact storage remains a formidable task. To address these challenges, we propose Spacetime Gaussian Feature Splatting as a novel dynamic scene representation, composed of three pivotal components. First, we formulate expressive Spacetime Gaussians by enhancing 3D Gaussians with temporal opacity and parametric motion/rotation. This enables Spacetime Gaussians to capture static, dynamic, as well as transient content within a scene. Second, we introduce splatted feature rendering, which replaces spherical harmonics with neural features. These features facilitate the modeling of view- and time-dependent appearance while maintaining small size. Third, we leverage the guidance of training error and coarse depth to sample new Gaussians in areas that are challenging to converge with existing pipelines. Experiments on several established real-world datasets demonstrate that our method achieves state-of-the-art rendering quality and speed, while retaining compact storage. At 8K resolution, our lite-version model can render at 60 FPS on an Nvidia RTX 4090 GPU. Our code is available at https://github.com/oppo-us-research/SpacetimeGaussians.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.16812v2.pdf)

## Surface Representation
#### Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

**Authors**: Bo Dai, Dahua Lin, LiMin Wang, Yuanbo Xiangli, Linning Xu, Mulin Yu, Tao Lu

<details>
<summary><b>Abstract</b></summary>
Neural rendering methods have significantly advanced photo-realistic 3D scene rendering in various academic and industrial applications. The recent 3D Gaussian Splatting method has achieved the state-of-the-art rendering quality and speed combining the benefits of both primitive-based representations and volumetric representations. However, it often leads to heavily redundant Gaussians that try to fit every training view, neglecting the underlying scene geometry. Consequently, the resulting model becomes less robust to significant view changes, texture-less area and lighting effects. We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum. Anchor growing and pruning strategies are developed based on the importance of neural Gaussians to reliably improve the scene coverage. We show that our method effectively reduces redundant Gaussians while delivering high-quality rendering. We also demonstrates an enhanced capability to accommodate scenes with varying levels-of-detail and view-dependent observations, without sacrificing the rendering speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00109v1.pdf)

#### 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

**Authors**: Shenghua Gao, Andreas Geiger, Anpei Chen, Zehao Yu, Binbin Huang

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has recently revolutionized radiance field reconstruction, achieving high quality novel view synthesis and fast rendering speed without baking. However, 3DGS fails to accurately represent surfaces due to the multi-view inconsistent nature of 3D Gaussians. We present 2D Gaussian Splatting (2DGS), a novel approach to model and reconstruct geometrically accurate radiance fields from multi-view images. Our key idea is to collapse the 3D volume into a set of 2D oriented planar Gaussian disks. Unlike 3D Gaussians, 2D Gaussians provide view-consistent geometry while modeling surfaces intrinsically. To accurately recover thin surfaces and achieve stable optimization, we introduce a perspective-correct 2D splatting process utilizing ray-splat intersection and rasterization. Additionally, we incorporate depth distortion and normal consistency terms to further enhance the quality of the reconstructions. We demonstrate that our differentiable renderer allows for noise-free and detailed geometry reconstruction while maintaining competitive appearance quality, fast training speed, and real-time rendering.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.17888v2.pdf)

#### SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

**Authors**: Vincent Lepetit, Antoine Guédon

<details>
<summary><b>Abstract</b></summary>
We propose a method to allow precise and extremely fast mesh extraction from 3D Gaussian Splatting. Gaussian Splatting has recently become very popular as it yields realistic rendering while being significantly faster to train than NeRFs. It is however challenging to extract a mesh from the millions of tiny 3D gaussians as these gaussians tend to be unorganized after optimization and no method has been proposed so far. Our first key contribution is a regularization term that encourages the gaussians to align well with the surface of the scene. We then introduce a method that exploits this alignment to extract a mesh from the Gaussians using Poisson reconstruction, which is fast, scalable, and preserves details, in contrast to the Marching Cubes algorithm usually applied to extract meshes from Neural SDFs. Finally, we introduce an optional refinement strategy that binds gaussians to the surface of the mesh, and jointly optimizes these Gaussians and the mesh through Gaussian splatting rendering. This enables easy editing, sculpting, rigging, animating, compositing and relighting of the Gaussians using traditional softwares by manipulating the mesh instead of the gaussians themselves. Retrieving such an editable mesh for realistic rendering is done within minutes with our method, compared to hours with the state-of-the-art methods on neural SDFs, while providing a better rendering quality. Our project page is the following: https://anttwo.github.io/sugar/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.12775v3.pdf)

#### GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction

**Authors**: Bo Dai, Yuanbo Xiangli, Lihan Jiang, Linning Xu, Tao Lu, Mulin Yu

<details>
<summary><b>Abstract</b></summary>
Presenting a 3D scene from multiview images remains a core and long-standing challenge in computer vision and computer graphics. Two main requirements lie in rendering and reconstruction. Notably, SOTA rendering quality is usually achieved with neural volumetric rendering techniques, which rely on aggregated point/primitive-wise color and neglect the underlying scene geometry. Learning of neural implicit surfaces is sparked from the success of neural rendering. Current works either constrain the distribution of density fields or the shape of primitives, resulting in degraded rendering quality and flaws on the learned scene surfaces. The efficacy of such methods is limited by the inherent constraints of the chosen neural representation, which struggles to capture fine surface details, especially for larger, more intricate scenes. To address these issues, we introduce GSDF, a novel dual-branch architecture that combines the benefits of a flexible and efficient 3D Gaussian Splatting (3DGS) representation with neural Signed Distance Fields (SDF). The core idea is to leverage and enhance the strengths of each branch while alleviating their limitation through mutual guidance and joint supervision. We show on diverse scenes that our design unlocks the potential for more accurate and detailed surface reconstructions, and at the meantime benefits 3DGS rendering with structures that are more aligned with the underlying geometry.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.16964v1.pdf)

#### 3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting

**Authors**: Xiaojuan Qi, Jiangmiao Pang, Yilun Chen, ZiYi Yang, Xiuzhe Wu, Yi-Hua Huang, Yang-tian Sun, Xiaoyang Lyu

<details>
<summary><b>Abstract</b></summary>
In this paper, we present an implicit surface reconstruction method with 3D Gaussian Splatting (3DGS), namely 3DGSR, that allows for accurate 3D reconstruction with intricate details while inheriting the high efficiency and rendering quality of 3DGS. The key insight is incorporating an implicit signed distance field (SDF) within 3D Gaussians to enable them to be aligned and jointly optimized. First, we introduce a differentiable SDF-to-opacity transformation function that converts SDF values into corresponding Gaussians' opacities. This function connects the SDF and 3D Gaussians, allowing for unified optimization and enforcing surface constraints on the 3D Gaussians. During learning, optimizing the 3D Gaussians provides supervisory signals for SDF learning, enabling the reconstruction of intricate details. However, this only provides sparse supervisory signals to the SDF at locations occupied by Gaussians, which is insufficient for learning a continuous SDF. Then, to address this limitation, we incorporate volumetric rendering and align the rendered geometric attributes (depth, normal) with those derived from 3D Gaussians. This consistency regularization introduces supervisory signals to locations not covered by discrete 3D Gaussians, effectively eliminating redundant surfaces outside the Gaussian sampling range. Our extensive experimental results demonstrate that our 3DGSR method enables high-quality 3D surface reconstruction while preserving the efficiency and rendering quality of 3DGS. Besides, our method competes favorably with leading surface reconstruction techniques while offering a more efficient learning process and much better rendering qualities. The code will be available at https://github.com/CVMI-Lab/3DGSR.
</details>
[📄 Paper](https://arxiv.org/pdf/2404.00409v1.pdf)

#### High-quality Surface Reconstruction using Gaussian Surfels

**Authors**: Weiwei Xu, Huamin Wang, Xinguo Liu, Wenxiang Xie, Jiamin Xu, Pinxuan Dai

<details>
<summary><b>Abstract</b></summary>
We propose a novel point-based representation, Gaussian surfels, to combine the advantages of the flexible optimization procedure in 3D Gaussian points and the surface alignment property of surfels. This is achieved by directly setting the z-scale of 3D Gaussian points to 0, effectively flattening the original 3D ellipsoid into a 2D ellipse. Such a design provides clear guidance to the optimizer. By treating the local z-axis as the normal direction, it greatly improves optimization stability and surface alignment. While the derivatives to the local z-axis computed from the covariance matrix are zero in this setting, we design a self-supervised normal-depth consistency loss to remedy this issue. Monocular normal priors and foreground masks are incorporated to enhance the quality of the reconstruction, mitigating issues related to highlights and background. We propose a volumetric cutting method to aggregate the information of Gaussian surfels so as to remove erroneous points in depth maps generated by alpha blending. Finally, we apply screened Poisson reconstruction method to the fused depth maps to extract the surface mesh. Experimental results show that our method demonstrates superior performance in surface reconstruction compared to state-of-the-art neural volume rendering and point-based rendering methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.17774v2.pdf)

#### Implicit Geometric Regularization for Learning Shapes

**Authors**: Niv Haim, Yaron Lipman, Matan Atzmon, Lior Yariv, Amos Gropp

<details>
<summary><b>Abstract</b></summary>
Representing shapes as level sets of neural networks has been recently proved to be useful for different shape analysis and reconstruction tasks. So far, such representations were computed using either: (i) pre-computed implicit shape representations; or (ii) loss functions explicitly defined over the neural level sets. In this paper we offer a new paradigm for computing high fidelity implicit neural representations directly from raw data (i.e., point clouds, with or without normal information). We observe that a rather simple loss function, encouraging the neural network to vanish on the input point cloud and to have a unit norm gradient, possesses an implicit geometric regularization property that favors smooth and natural zero level set surfaces, avoiding bad zero-loss solutions. We provide a theoretical analysis of this property for the linear case, and show that, in practice, our method leads to state of the art implicit neural representations with higher level-of-details and fidelity compared to previous methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2002.10099v2.pdf)

#### NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance

**Authors**: Gim Hee Lee, Chen Li, Hanlin Chen

<details>
<summary><b>Abstract</b></summary>
Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00846v1.pdf)

#### NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

**Authors**: Wenping Wang, Taku Komura, Christian Theobalt, YuAn Liu, Lingjie Liu, Peng Wang

<details>
<summary><b>Abstract</b></summary>
We present a novel neural surface reconstruction method, called NeuS, for reconstructing objects and scenes with high fidelity from 2D image inputs. Existing neural surface reconstruction approaches, such as DVR and IDR, require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures. Meanwhile, recent neural methods for novel view synthesis, such as NeRF and its variants, use volume rendering to produce a neural scene representation with robustness of optimization, even for highly complex objects. However, extracting high-quality surfaces from this learned implicit representation is difficult because there are not sufficient surface constraints in the representation. In NeuS, we propose to represent a surface as the zero-level set of a signed distance function (SDF) and develop a new volume rendering method to train a neural SDF representation. We observe that the conventional volume rendering method causes inherent geometric errors (i.e. bias) for surface reconstruction, and therefore propose a new formulation that is free of bias in the first order of approximation, thus leading to more accurate surface reconstruction even without the mask supervision. Experiments on the DTU dataset and the BlendedMVS dataset show that NeuS outperforms the state-of-the-arts in high-quality surface reconstruction, especially for objects and scenes with complex structures and self-occlusion.
</details>

[📄 Paper](https://arxiv.org/pdf/2106.10689v3.pdf)

#### Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes

**Authors**: Andreas Geiger, Torsten Sattler, Zehao Yu

<details>
<summary><b>Abstract</b></summary>
Recently, 3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results, while allowing the rendering of high-resolution images in real-time. However, leveraging 3D Gaussians for surface reconstruction poses significant challenges due to the explicit and disconnected nature of 3D Gaussians. In this work, we present Gaussian Opacity Fields (GOF), a novel approach for efficient, high-quality, and compact surface reconstruction in unbounded scenes. Our GOF is derived from ray-tracing-based volume rendering of 3D Gaussians, enabling direct geometry extraction from 3D Gaussians by identifying its levelset, without resorting to Poisson reconstruction or TSDF fusion as in previous work. We approximate the surface normal of Gaussians as the normal of the ray-Gaussian intersection plane, enabling the application of regularization that significantly enhances geometry. Furthermore, we develop an efficient geometry extraction method utilizing marching tetrahedra, where the tetrahedral grids are induced from 3D Gaussians and thus adapt to the scene's complexity. Our evaluations reveal that GOF surpasses existing 3DGS-based methods in surface reconstruction and novel view synthesis. Further, it compares favorably to, or even outperforms, neural implicit methods in both quality and speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.10772v1.pdf)

## Editable 3D Gaussian Splatting
### Manipulation by Text
#### GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

**Authors**: Guosheng Lin, Huaping Liu, Lei Yang, Zhongang Cai, Yikai Wang, Xiaofeng Yang, Feng Wang, Chi Zhang, Zilong Chen, YiWen Chen

<details>
<summary><b>Abstract</b></summary>
3D editing plays a crucial role in many areas such as gaming and virtual reality. Traditional 3D editing methods, which rely on representations like meshes and point clouds, often fall short in realistically depicting complex scenes. On the other hand, methods based on implicit 3D representations, like Neural Radiance Field (NeRF), render complex scenes effectively but suffer from slow processing speeds and limited control over specific scene areas. In response to these challenges, our paper presents GaussianEditor, an innovative and efficient 3D editing algorithm based on Gaussian Splatting (GS), a novel 3D representation. GaussianEditor enhances precision and control in editing through our proposed Gaussian semantic tracing, which traces the editing target throughout the training process. Additionally, we propose Hierarchical Gaussian splatting (HGS) to achieve stabilized and fine results under stochastic generative guidance from 2D diffusion models. We also develop editing strategies for efficient object removal and integration, a challenging task for existing methods. Our comprehensive experiments demonstrate GaussianEditor's superior control, efficacy, and rapid performance, marking a significant advancement in 3D editing. Project Page: https://buaacyw.github.io/gaussian-editor/
</details>
[📄 Paper](https://arxiv.org/pdf/2311.14521v4.pdf)

#### GaussianEditor: Editing 3D Gaussians Delicately with Text Instructions

**Authors**: Qi Tian, Lingxi Xie, Xiaopeng Zhang, Junjie Wang, Jiemin Fang

<details>
<summary><b>Abstract</b></summary>
Recently, impressive results have been achieved in 3D scene editing with text instructions based on a 2D diffusion model. However, current diffusion models primarily generate images by predicting noise in the latent space, and the editing is usually applied to the whole image, which makes it challenging to perform delicate, especially localized, editing for 3D scenes. Inspired by recent 3D Gaussian splatting, we propose a systematic framework, named GaussianEditor, to edit 3D scenes delicately via 3D Gaussians with text instructions. Benefiting from the explicit property of 3D Gaussians, we design a series of techniques to achieve delicate editing. Specifically, we first extract the region of interest (RoI) corresponding to the text instruction, aligning it to 3D Gaussians. The Gaussian RoI is further used to control the editing process. Our framework can achieve more delicate and precise editing of 3D scenes than previous methods while enjoying much faster training speed, i.e. within 20 minutes on a single V100 GPU, more than twice as fast as Instruct-NeRF2NeRF (45 minutes -- 2 hours).
</details>
[📄 Paper](https://arxiv.org/pdf/2311.16037v1.pdf)

#### DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation

**Authors**: Gang Zeng, Ziwei Liu, Hang Zhou, Jiawei Ren, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
Recent advances in 3D content creation mostly leverage optimization-based 3D generation via score distillation sampling (SDS). Though promising results have been exhibited, these methods often suffer from slow per-sample optimization, limiting their practical usage. In this paper, we propose DreamGaussian, a novel 3D content generation framework that achieves both efficiency and quality simultaneously. Our key insight is to design a generative 3D Gaussian Splatting model with companioned mesh extraction and texture refinement in UV space. In contrast to the occupancy pruning used in Neural Radiance Fields, we demonstrate that the progressive densification of 3D Gaussians converges significantly faster for 3D generative tasks. To further enhance the texture quality and facilitate downstream applications, we introduce an efficient algorithm to convert 3D Gaussians into textured meshes and apply a fine-tuning stage to refine the details. Extensive experiments demonstrate the superior efficiency and competitive generation quality of our proposed approach. Notably, DreamGaussian produces high-quality textured meshes in just 2 minutes from a single-view image, achieving approximately 10 times acceleration compared to existing methods.
</details>
[📄 Paper](https://arxiv.org/pdf/2309.16653v2.pdf)

#### GSEdit: Efficient Text-Guided Editing of 3D Objects via Gaussian Splatting

**Authors**: Emanuele Rodolà, Daniele Baieri, Andrea Sanchietti, Francesco Palandra

<details>
<summary><b>Abstract</b></summary>
We present GSEdit, a pipeline for text-guided 3D object editing based on Gaussian Splatting models. Our method enables the editing of the style and appearance of 3D objects without altering their main details, all in a matter of minutes on consumer hardware. We tackle the problem by leveraging Gaussian splatting to represent 3D scenes, and we optimize the model while progressively varying the image supervision by means of a pretrained image-based diffusion model. The input object may be given as a 3D triangular mesh, or directly provided as Gaussians from a generative model such as DreamGaussian. GSEdit ensures consistency across different viewpoints, maintaining the integrity of the original object's information. Compared to previously proposed methods relying on NeRF-like MLP models, GSEdit stands out for its efficiency, making 3D editing tasks much faster. Our editing process is refined via the application of the SDS loss, ensuring that our edits are both precise and accurate. Our comprehensive evaluation demonstrates that GSEdit effectively alters object shape and appearance following the given textual instructions while preserving their coherence and detail.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.05154v2.pdf)

#### View-Consistent 3D Editing with Gaussian Splatting

**Authors**: Hanwang Zhang, Long Chen, Na Zhao, Zike Wu, Xuanyu Yi, Yuxuan Wang

<details>
<summary><b>Abstract</b></summary>
The advent of 3D Gaussian Splatting (3DGS) has revolutionized 3D editing, offering efficient, high-fidelity rendering and enabling precise local manipulations. Currently, diffusion-based 2D editing models are harnessed to modify multi-view rendered images, which then guide the editing of 3DGS models. However, this approach faces a critical issue of multi-view inconsistency, where the guidance images exhibit significant discrepancies across views, leading to mode collapse and visual artifacts of 3DGS. To this end, we introduce View-consistent Editing (VcEdit), a novel framework that seamlessly incorporates 3DGS into image editing processes, ensuring multi-view consistency in edited guidance images and effectively mitigating mode collapse issues. VcEdit employs two innovative consistency modules: the Cross-attention Consistency Module and the Editing Consistency Module, both designed to reduce inconsistencies in edited images. By incorporating these consistency modules into an iterative pattern, VcEdit proficiently resolves the issue of multi-view inconsistency, facilitating high-quality 3DGS editing across a diverse range of scenes. Further code and video results are released at http://yuxuanw.me/vcedit/.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11868v6.pdf)

#### GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

**Authors**: Victor Adrian Prisacariu, Philip Torr, Ian Reid, Guangrun Wang, Xinghui Li, Jia-Wang Bian, Jing Wu

<details>
<summary><b>Abstract</b></summary>
We propose GaussCtrl, a text-driven method to edit a 3D scene reconstructed by the 3D Gaussian Splatting (3DGS). Our method first renders a collection of images by using the 3DGS and edits them by using a pre-trained 2D diffusion model (ControlNet) based on the input prompt, which is then used to optimise the 3D model. Our key contribution is multi-view consistent editing, which enables editing all images together instead of iteratively editing one image while updating the 3D model as in previous works. It leads to faster editing as well as higher visual quality. This is achieved by the two terms: (a) depth-conditioned editing that enforces geometric consistency across multi-view images by leveraging naturally consistent depth maps. (b) attention-based latent code alignment that unifies the appearance of edited images by conditioning their editing to several reference views through self and cross-view attention between images' latent representations. Experiments demonstrate that our method achieves faster editing and better visual results than previous state-of-the-art methods.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.08733v4.pdf)

### Manipulation by Other Conditions
#### Point'n Move: Interactive Scene Object Manipulation on Gaussian Splatting Radiance Fields

**Authors**: Hongchuan Yu, Jiajun Huang

<details>
<summary><b>Abstract</b></summary>
We propose Point'n Move, a method that achieves interactive scene object manipulation with exposed region inpainting. Interactivity here further comes from intuitive object selection and real-time editing. To achieve this, we adopt Gaussian Splatting Radiance Field as the scene representation and fully leverage its explicit nature and speed advantage. Its explicit representation formulation allows us to devise a 2D prompt points to 3D mask dual-stage self-prompting segmentation algorithm, perform mask refinement and merging, minimize change as well as provide good initialization for scene inpainting and perform editing in real-time without per-editing training, all leads to superior quality and performance. We test our method by performing editing on both forward-facing and 360 scenes. We also compare our method against existing scene object removal methods, showing superior quality despite being more capable and having a speed advantage.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16737v1.pdf)

#### TIP-Editor: An Accurate 3D Editor Following Both Text-Prompts And Image-Prompts

**Authors**: Ying Shan, Liang Lin, Guanbin Li, Yan-Pei Cao, Di Kang, Jingyu Zhuang

<details>
<summary><b>Abstract</b></summary>
Text-driven 3D scene editing has gained significant attention owing to its convenience and user-friendliness. However, existing methods still lack accurate control of the specified appearance and location of the editing result due to the inherent limitations of the text description. To this end, we propose a 3D scene editing framework, TIPEditor, that accepts both text and image prompts and a 3D bounding box to specify the editing region. With the image prompt, users can conveniently specify the detailed appearance/style of the target content in complement to the text description, enabling accurate control of the appearance. Specifically, TIP-Editor employs a stepwise 2D personalization strategy to better learn the representation of the existing scene and the reference image, in which a localization loss is proposed to encourage correct object placement as specified by the bounding box. Additionally, TIPEditor utilizes explicit and flexible 3D Gaussian splatting as the 3D representation to facilitate local editing while keeping the background unchanged. Extensive experiments have demonstrated that TIP-Editor conducts accurate editing following the text and image prompts in the specified bounding box region, consistently outperforming the baselines in editing quality, and the alignment to the prompts, qualitatively and quantitatively.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.14828v3.pdf)

### Stylization
#### Gaussian Splatting in Style

**Authors**: Daniel Cremers, Tarun Yenamandra, Cecilia Curreli, Mariia Gladkova, Abhishek Saroha

<details>
<summary><b>Abstract</b></summary>
Scene stylization extends the work of neural style transfer to three spatial dimensions. A vital challenge in this problem is to maintain the uniformity of the stylized appearance across a multi-view setting. A vast majority of the previous works achieve this by optimizing the scene with a specific style image. In contrast, we propose a novel architecture trained on a collection of style images, that at test time produces high quality stylized novel views. Our work builds up on the framework of 3D Gaussian splatting. For a given scene, we take the pretrained Gaussians and process them using a multi resolution hash grid and a tiny MLP to obtain the conditional stylised views. The explicit nature of 3D Gaussians give us inherent advantages over NeRF-based methods including geometric consistency, along with having a fast training and rendering regime. This enables our method to be useful for vast practical use cases such as in augmented or virtual reality applications. Through our experiments, we show our methods achieve state-of-the-art performance with superior visual quality on various indoor and outdoor real-world data.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.08498v1.pdf)

### Animation
#### CoGS: Controllable Gaussian Splatting

**Authors**: László A. Jeni, Koichiro Niinuma, Zoltán Á. Milacski, Joel Julin, Heng Yu

<details>
<summary><b>Abstract</b></summary>
Capturing and re-animating the 3D structure of articulated objects present significant barriers. On one hand, methods requiring extensively calibrated multi-view setups are prohibitively complex and resource-intensive, limiting their practical applicability. On the other hand, while single-camera Neural Radiance Fields (NeRFs) offer a more streamlined approach, they have excessive training and rendering costs. 3D Gaussian Splatting would be a suitable alternative but for two reasons. Firstly, existing methods for 3D dynamic Gaussians require synchronized multi-view cameras, and secondly, the lack of controllability in dynamic scenarios. We present CoGS, a method for Controllable Gaussian Splatting, that enables the direct manipulation of scene elements, offering real-time control of dynamic scenes without the prerequisite of pre-computing control signals. We evaluated CoGS using both synthetic and real-world datasets that include dynamic objects that differ in degree of difficulty. In our evaluations, CoGS consistently outperformed existing dynamic and controllable neural representations in terms of visual fidelity.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.05664v2.pdf)

#### BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors

**Authors**: Baoquan Chen, Libin Liu, Weiyu Li, Qingzhe Gao, Tingyang Zhang

<details>
<summary><b>Abstract</b></summary>
Animatable 3D reconstruction has significant applications across various fields, primarily relying on artists' handcraft creation. Recently, some studies have successfully constructed animatable 3D models from monocular videos. However, these approaches require sufficient view coverage of the object within the input video and typically necessitate significant time and computational costs for training and rendering. This limitation restricts the practical applications. In this work, we propose a method to build animatable 3D Gaussian Splatting from monocular video with diffusion priors. The 3D Gaussian representations significantly accelerate the training and rendering process, and the diffusion priors allow the method to learn 3D models with limited viewpoints. We also present the rigid regularization to enhance the utilization of the priors. We perform an extensive evaluation across various real-world videos, demonstrating its superior performance compared to the current state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11427v1.pdf)

#### SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes

**Authors**: Xiaojuan Qi, Yan-Pei Cao, Xiaoyang Lyu, ZiYi Yang, Yang-tian Sun, Yi-Hua Huang

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis for dynamic scenes is still a challenging problem in computer vision and graphics. Recently, Gaussian splatting has emerged as a robust technique to represent static scenes and enable high-quality and real-time novel view synthesis. Building upon this technique, we propose a new representation that explicitly decomposes the motion and appearance of dynamic scenes into sparse control points and dense Gaussians, respectively. Our key idea is to use sparse control points, significantly fewer in number than the Gaussians, to learn compact 6 DoF transformation bases, which can be locally interpolated through learned interpolation weights to yield the motion field of 3D Gaussians. We employ a deformation MLP to predict time-varying 6 DoF transformations for each control point, which reduces learning complexities, enhances learning abilities, and facilitates obtaining temporal and spatial coherent motion patterns. Then, we jointly learn the 3D Gaussians, the canonical space locations of control points, and the deformation MLP to reconstruct the appearance, geometry, and dynamics of 3D scenes. During learning, the location and number of control points are adaptively adjusted to accommodate varying motion complexities in different regions, and an ARAP loss following the principle of as rigid as possible is developed to enforce spatial continuity and local rigidity of learned motions. Finally, thanks to the explicit sparse motion representation and its decomposition from appearance, our method can enable user-controlled motion editing while retaining high-fidelity appearances. Extensive experiments demonstrate that our approach outperforms existing approaches on novel view synthesis with a high rendering speed and enables novel appearance-preserved motion editing applications. Project page: https://yihua7.github.io/SC-GS-web/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.14937v3.pdf)

## Semantic Understanding
#### Gaussian Grouping: Segment and Edit Anything in 3D Scenes

**Authors**: Lei Ke, Fisher Yu, Martin Danelljan, Mingqiao Ye

<details>
<summary><b>Abstract</b></summary>
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by Segment Anything Model (SAM), along with introduced 3D spatial consistency regularization. Compared to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization, style transfer and scene recomposition. Our code and models are at https://github.com/lkeab/gaussian-grouping.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00732v2.pdf)

#### Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding

**Authors**: Shao-Hua Guan, Hao-Bin Duan, Miao Wang, Jin-Chuan Shi

<details>
<summary><b>Abstract</b></summary>
Open-vocabulary querying in 3D space is challenging but essential for scene understanding tasks such as object localization and segmentation. Language-embedded scene representations have made progress by incorporating language features into 3D spaces. However, their efficacy heavily depends on neural networks that are resource-intensive in training and rendering. Although recent 3D Gaussians offer efficient and high-quality novel view synthesis, directly embedding language features in them leads to prohibitive memory usage and decreased performance. In this work, we introduce Language Embedded 3D Gaussians, a novel scene representation for open-vocabulary query tasks. Instead of embedding high-dimensional raw semantic features on 3D Gaussians, we propose a dedicated quantization scheme that drastically alleviates the memory requirement, and a novel embedding procedure that achieves smoother yet high accuracy query, countering the multi-view feature inconsistencies and the high-frequency inductive bias in point-based representations. Our comprehensive experiments show that our representation achieves the best visual quality and language querying accuracy across current language-embedded representations, while maintaining real-time rendering frame rates on a single desktop GPU.
</details>
[📄 Paper](https://arxiv.org/pdf/2311.18482v1.pdf)

#### FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding

**Authors**: Mingyang Li, Yan Di, Yunwen Zhou, Pouya Samangouei, Xingxing Zuo

<details>
<summary><b>Abstract</b></summary>
Precisely perceiving the geometric and semantic properties of real-world 3D objects is crucial for the continued evolution of augmented reality and robotic applications. To this end, we present Foundation Model Embedded Gaussian Splatting (FMGS), which incorporates vision-language embeddings of foundation models into 3D Gaussian Splatting (GS). The key contribution of this work is an efficient method to reconstruct and represent 3D vision-language models. This is achieved by distilling feature maps generated from image-based foundation models into those rendered from our 3D model. To ensure high-quality rendering and fast training, we introduce a novel scene representation by integrating strengths from both GS and multi-resolution hash encodings (MHE). Our effective training procedure also introduces a pixel alignment loss that makes the rendered feature distance of the same semantic entities close, following the pixel-level semantic boundaries. Our results demonstrate remarkable multi-view semantic consistency, facilitating diverse downstream tasks, beating state-of-the-art methods by 10.2 percent on open-vocabulary language-based object detection, despite that we are 851X faster for inference. This research explores the intersection of vision, language, and 3D scene representation, paving the way for enhanced scene understanding in uncontrolled real-world environments. We plan to release the code on the project page.
</details>
[📄 Paper](https://arxiv.org/pdf/2401.01970v2.pdf)

#### Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields

**Authors**: Achuta Kadambi, Zhangyang Wang, Suya You, Pradyumna Chari, Dejia Xu, Zehao Zhu, Zhiwen Fan, Sicheng Jiang, Haoran Chang, Shijie Zhou

<details>
<summary><b>Abstract</b></summary>
3D scene representations have gained immense popularity in recent years. Methods that use Neural Radiance fields are versatile for traditional tasks such as novel view synthesis. In recent times, some work has emerged that aims to extend the functionality of NeRF beyond view synthesis, for semantically aware tasks such as editing and segmentation using 3D feature field distillation from 2D foundation models. However, these methods have two major limitations: (a) they are limited by the rendering speed of NeRF pipelines, and (b) implicitly represented feature fields suffer from continuity artifacts reducing feature quality. Recently, 3D Gaussian Splatting has shown state-of-the-art performance on real-time radiance field rendering. In this work, we go one step further: in addition to radiance field rendering, we enable 3D Gaussian splatting on arbitrary-dimension semantic features via 2D foundation model distillation. This translation is not straightforward: naively incorporating feature fields in the 3DGS framework encounters significant challenges, notably the disparities in spatial resolution and channel consistency between RGB images and feature maps. We propose architectural and training changes to efficiently avert this problem. Our proposed method is general, and our experiments showcase novel view semantic segmentation, language-guided editing and segment anything through learning feature fields from state-of-the-art 2D foundation models such as SAM and CLIP-LSeg. Across experiments, our distillation method is able to provide comparable or better results, while being significantly faster to both train and render. Additionally, to the best of our knowledge, we are the first method to enable point and bounding-box prompting for radiance field manipulation, by leveraging the SAM model. Project website at: https://feature-3dgs.github.io/
</details>
[📄 Paper](https://arxiv.org/pdf/2312.03203v3.pdf)

#### Emerging Properties in Self-Supervised Vision Transformers

**Authors**: Armand Joulin, Piotr Bojanowski, Julien Mairal, Hervé Jégou, Ishan Misra, Hugo Touvron, Mathilde Caron

<details>
<summary><b>Abstract</b></summary>
In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, self-supervised ViT features contain explicit information about the semantic segmentation of an image, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, these features are also excellent k-NN classifiers, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of momentum encoder, multi-crop training, and the use of small patches with ViTs. We implement our findings into a simple self-supervised method, called DINO, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.
</details>

[📄 Paper](https://arxiv.org/pdf/2104.14294v2.pdf)

#### LangSplat: 3D Language Gaussian Splatting

**Authors**: Hanspeter Pfister, Haoqian Wang, Jiawei Zhou, Wanhua Li, Minghan Qin

<details>
<summary><b>Abstract</b></summary>
Humans live in a 3D world and commonly use natural language to interact with a 3D scene. Modeling a 3D language field to support open-ended language queries in 3D has gained increasing attention recently. This paper introduces LangSplat, which constructs a 3D language field that enables precise and efficient open-vocabulary querying within 3D spaces. Unlike existing methods that ground CLIP language embeddings in a NeRF model, LangSplat advances the field by utilizing a collection of 3D Gaussians, each encoding language features distilled from CLIP, to represent the language field. By employing a tile-based splatting technique for rendering language features, we circumvent the costly rendering process inherent in NeRF. Instead of directly learning CLIP embeddings, LangSplat first trains a scene-wise language autoencoder and then learns language features on the scene-specific latent space, thereby alleviating substantial memory demands imposed by explicit modeling. Existing methods struggle with imprecise and vague 3D language fields, which fail to discern clear boundaries between objects. We delve into this issue and propose to learn hierarchical semantics using SAM, thereby eliminating the need for extensively querying the language field across various scales and the regularization of DINO features. Extensive experimental results show that LangSplat significantly outperforms the previous state-of-the-art method LERF by a large margin. Notably, LangSplat is extremely efficient, achieving a 199 $\times$ speedup compared to LERF at the resolution of 1440 $\times$ 1080. We strongly recommend readers to check out our video results at https://langsplat.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.16084v2.pdf)

#### CoSSegGaussians: Compact and Swift Scene Segmenting 3D Gaussians with Dual Feature Fusion

**Authors**: Zejian yuan, Zhaohui Wang, Yongjia Ma, Tianyu Zhang, Bin Dou

<details>
<summary><b>Abstract</b></summary>
We propose Compact and Swift Segmenting 3D Gaussians(CoSSegGaussians), a method for compact 3D-consistent scene segmentation at fast rendering speed with only RGB images input. Previous NeRF-based segmentation methods have relied on time-consuming neural scene optimization. While recent 3D Gaussian Splatting has notably improved speed, existing Gaussian-based segmentation methods struggle to produce compact masks, especially in zero-shot segmentation. This issue probably stems from their straightforward assignment of learnable parameters to each Gaussian, resulting in a lack of robustness against cross-view inconsistent 2D machine-generated labels. Our method aims to address this problem by employing Dual Feature Fusion Network as Gaussians' segmentation field. Specifically, we first optimize 3D Gaussians under RGB supervision. After Gaussian Locating, DINO features extracted from images are applied through explicit unprojection, which are further incorporated with spatial features from the efficient point cloud processing network. Feature aggregation is utilized to fuse them in a global-to-local strategy for compact segmentation features. Experimental results show that our model outperforms baselines on both semantic and panoptic zero-shot segmentation task, meanwhile consumes less than 10% inference time compared to NeRF-based methods. Code and more results will be available at https://David-Dou.github.io/CoSSegGaussians
</details>

[📄 Paper](https://arxiv.org/pdf/2401.05925v3.pdf)

#### 2D-Guided 3D Gaussian Segmentation

**Authors**: Pengyuan Zhou, Lin Wang, Yong Liao, Wenjun Wu, Haolin Shi, Haoran Li, Kun Lan

<details>
<summary><b>Abstract</b></summary>
Recently, 3D Gaussian, as an explicit 3D representation method, has demonstrated strong competitiveness over NeRF (Neural Radiance Fields) in terms of expressing complex scenes and training duration. These advantages signal a wide range of applications for 3D Gaussians in 3D understanding and editing. Meanwhile, the segmentation of 3D Gaussians is still in its infancy. The existing segmentation methods are not only cumbersome but also incapable of segmenting multiple objects simultaneously in a short amount of time. In response, this paper introduces a 3D Gaussian segmentation method implemented with 2D segmentation as supervision. This approach uses input 2D segmentation maps to guide the learning of the added 3D Gaussian semantic information, while nearest neighbor clustering and statistical filtering refine the segmentation results. Experiments show that our concise method can achieve comparable performances on mIOU and mAcc for multi-object segmentation as previous single-object segmentation methods.
</details>
[📄 Paper](https://arxiv.org/pdf/2312.16047v1.pdf)

## Physics Simulation
#### 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

**Authors**: Xinggang Wang, Qi Tian, Wenyu Liu, Wei Wei, Xiaopeng Zhang, Lingxi Xie, Jiemin Fang, Taoran Yi, Guanjun Wu

<details>
<summary><b>Abstract</b></summary>
Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800$\times$800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods. More demos and code are available at https://guanjunwu.github.io/4dgs/.
</details>

[📄 Paper](https://arxiv.org/pdf/2310.08528v3.pdf)

#### Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis

**Authors**: Deva Ramanan, Bastian Leibe, Georgios Kopanas, Jonathon Luiten

<details>
<summary><b>Abstract</b></summary>
We present a method that simultaneously addresses the tasks of dynamic scene novel-view synthesis and six degree-of-freedom (6-DOF) tracking of all dense scene elements. We follow an analysis-by-synthesis framework, inspired by recent work that models scenes as a collection of 3D Gaussians which are optimized to reconstruct input images via differentiable rendering. To model dynamic scenes, we allow Gaussians to move and rotate over time while enforcing that they have persistent color, opacity, and size. By regularizing Gaussians' motion and rotation with local-rigidity constraints, we show that our Dynamic 3D Gaussians correctly model the same area of physical space over time, including the rotation of that space. Dense 6-DOF tracking and dynamic reconstruction emerges naturally from persistent dynamic view synthesis, without requiring any correspondence or flow as input. We demonstrate a large number of downstream applications enabled by our representation, including first-person view synthesis, dynamic compositional scene synthesis, and 4D video editing.
</details>
[📄 Paper](https://arxiv.org/pdf/2308.09713v1.pdf)

#### PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics

**Authors**: Yuxing Qiu, Chenfanfu Jiang, Yin Yang, Yutao Feng, Xuan Li, Zeshun Zong, Tianyi Xie

<details>
<summary><b>Abstract</b></summary>
We introduce PhysGaussian, a new method that seamlessly integrates physically grounded Newtonian dynamics within 3D Gaussians to achieve high-quality novel motion synthesis. Employing a custom Material Point Method (MPM), our approach enriches 3D Gaussian kernels with physically meaningful kinematic deformation and mechanical stress attributes, all evolved in line with continuum mechanics principles. A defining characteristic of our method is the seamless integration between physical simulation and visual rendering: both components utilize the same 3D Gaussian kernels as their discrete representations. This negates the necessity for triangle/tetrahedron meshing, marching cubes, "cage meshes," or any other geometry embedding, highlighting the principle of "what you see is what you simulate (WS$^2$)." Our method demonstrates exceptional versatility across a wide variety of materials--including elastic entities, metals, non-Newtonian fluids, and granular materials--showcasing its strong capabilities in creating diverse visual content with novel viewpoints and movements. Our project page is at: https://xpandora.github.io/PhysGaussian/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.12198v3.pdf)

# Technical Classification
## Initialization
#### Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting

**Authors**: Seungryong Kim, Seonghoon Park, Jiwon Kang, Honggyu An, Jisang Han, Jaewoo Jung

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting (3DGS) has recently demonstrated impressive capabilities in real-time novel view synthesis and 3D reconstruction. However, 3DGS heavily depends on the accurate initialization derived from Structure-from-Motion (SfM) methods. When the quality of the initial point cloud deteriorates, such as in the presence of noise or when using randomly initialized point cloud, 3DGS often undergoes large performance drops. To address this limitation, we propose a novel optimization strategy dubbed RAIN-GS (Relaing Accurate Initialization Constraint for 3D Gaussian Splatting). Our approach is based on an in-depth analysis of the original 3DGS optimization scheme and the analysis of the SfM initialization in the frequency domain. Leveraging simple modifications based on our analyses, RAIN-GS successfully trains 3D Gaussians from sub-optimal point cloud (e.g., randomly initialized point cloud), effectively relaxing the need for accurate initialization. We demonstrate the efficacy of our strategy through quantitative and qualitative comparisons on multiple datasets, where RAIN-GS trained with random point cloud achieves performance on-par with or even better than 3DGS trained with accurate SfM point cloud. Our project page and code can be found at https://ku-cvlab.github.io/RAIN-GS.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.09413v2.pdf)

#### Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction

**Authors**: Jan Eric Lenssen, Eddy Ilg, Raza Yunus, Christopher Wewer, Devikalyan Das

<details>
<summary><b>Abstract</b></summary>
Reconstructing dynamic objects from monocular videos is a severely underconstrained and challenging problem, and recent work has approached it in various directions. However, owing to the ill-posed nature of this problem, there has been no solution that can provide consistent, high-quality novel views from camera positions that are significantly different from the training views. In this work, we introduce Neural Parametric Gaussians (NPGs) to take on this challenge by imposing a two-stage approach: first, we fit a low-rank neural deformation model, which then is used as regularization for non-rigid reconstruction in the second stage. The first stage learns the object's deformations such that it preserves consistency in novel views. The second stage obtains high reconstruction quality by optimizing 3D Gaussians that are driven by the coarse model. To this end, we introduce a local 3D Gaussian representation, where temporally shared Gaussians are anchored in and deformed by local oriented volumes. The resulting combined model can be rendered as radiance fields, resulting in high-quality photo-realistic reconstructions of the non-rigidly deforming objects. We demonstrate that NPGs achieve superior results compared to previous works, especially in challenging scenarios with few multi-view cues.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.01196v2.pdf)

#### Text-to-3D using Gaussian Splatting

**Authors**: Huaping Liu, Yikai Wang, Feng Wang, Zilong Chen

<details>
<summary><b>Abstract</b></summary>
Automatic text-to-3D generation that combines Score Distillation Sampling (SDS) with the optimization of volume rendering has achieved remarkable progress in synthesizing realistic 3D objects. Yet most existing text-to-3D methods by SDS and volume rendering suffer from inaccurate geometry, e.g., the Janus issue, since it is hard to explicitly integrate 3D priors into implicit 3D representations. Besides, it is usually time-consuming for them to generate elaborate 3D models with rich colors. In response, this paper proposes GSGEN, a novel method that adopts Gaussian Splatting, a recent state-of-the-art representation, to text-to-3D generation. GSGEN aims at generating high-quality 3D objects and addressing existing shortcomings by exploiting the explicit nature of Gaussian Splatting that enables the incorporation of 3D prior. Specifically, our method adopts a progressive optimization strategy, which includes a geometry optimization stage and an appearance refinement stage. In geometry optimization, a coarse representation is established under 3D point cloud diffusion prior along with the ordinary 2D SDS optimization, ensuring a sensible and 3D-consistent rough shape. Subsequently, the obtained Gaussians undergo an iterative appearance refinement to enrich texture details. In this stage, we increase the number of Gaussians by compactness-based densification to enhance continuity and improve fidelity. With these designs, our approach can generate 3D assets with delicate details and accurate geometry. Extensive evaluations demonstrate the effectiveness of our method, especially for capturing high-frequency components. Our code is available at https://github.com/gsgen3d/gsgen
</details>

[📄 Paper](https://arxiv.org/pdf/2309.16585v4.pdf)

#### AGG: Amortized Generative 3D Gaussians for Single Image to 3D

**Authors**: Arash Vahdat, Zhangyang Wang, Jiaming Song, Sifei Liu, Morteza Mardani, Ye Yuan, Dejia Xu

<details>
<summary><b>Abstract</b></summary>
Given the growing need for automatic 3D content creation pipelines, various 3D representations have been studied to generate 3D objects from a single image. Due to its superior rendering efficiency, 3D Gaussian splatting-based models have recently excelled in both 3D reconstruction and generation. 3D Gaussian splatting approaches for image to 3D generation are often optimization-based, requiring many computationally expensive score-distillation steps. To overcome these challenges, we introduce an Amortized Generative 3D Gaussian framework (AGG) that instantly produces 3D Gaussians from a single image, eliminating the need for per-instance optimization. Utilizing an intermediate hybrid representation, AGG decomposes the generation of 3D Gaussian locations and other appearance attributes for joint optimization. Moreover, we propose a cascaded pipeline that first generates a coarse representation of the 3D data and later upsamples it with a 3D Gaussian super-resolution module. Our method is evaluated against existing optimization-based 3D Gaussian frameworks and sampling-based pipelines utilizing other 3D representations, where AGG showcases competitive generation abilities both qualitatively and quantitatively while being several orders of magnitude faster. Project page: https://ir1d.github.io/AGG/
</details>

[📄 Paper](https://arxiv.org/pdf/2401.04099v1.pdf)

#### GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting

**Authors**: Qi Tian, Wei Shen, Xiaopeng Zhang, Lingxi Xie, Ruofan Liang, Jiemin Fang, Sikuang Li, Chen Yang

<details>
<summary><b>Abstract</b></summary>
Reconstructing and rendering 3D objects from highly sparse views is of critical importance for promoting applications of 3D vision techniques and improving user experience. However, images from sparse views only contain very limited 3D information, leading to two significant challenges: 1) Difficulty in building multi-view consistency as images for matching are too few; 2) Partially omitted or highly compressed object information as view coverage is insufficient. To tackle these challenges, we propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting, that achieves high rendering quality with only 4 input images. We first introduce techniques of visual hull and floater elimination which explicitly inject structure priors into the initial optimization process for helping build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. Our GaussianObject is evaluated on several challenging datasets, including MipNeRF360, OmniObject3D, and OpenIllumination, achieving strong reconstruction results from only 4 views and significantly outperforming previous state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10259v2.pdf)

## Attribute Expansion
#### Neural Parametric Gaussians for Monocular Non-Rigid Object Reconstruction

**Authors**: Jan Eric Lenssen, Eddy Ilg, Raza Yunus, Christopher Wewer, Devikalyan Das

<details>
<summary><b>Abstract</b></summary>
Reconstructing dynamic objects from monocular videos is a severely underconstrained and challenging problem, and recent work has approached it in various directions. However, owing to the ill-posed nature of this problem, there has been no solution that can provide consistent, high-quality novel views from camera positions that are significantly different from the training views. In this work, we introduce Neural Parametric Gaussians (NPGs) to take on this challenge by imposing a two-stage approach: first, we fit a low-rank neural deformation model, which then is used as regularization for non-rigid reconstruction in the second stage. The first stage learns the object's deformations such that it preserves consistency in novel views. The second stage obtains high reconstruction quality by optimizing 3D Gaussians that are driven by the coarse model. To this end, we introduce a local 3D Gaussian representation, where temporally shared Gaussians are anchored in and deformed by local oriented volumes. The resulting combined model can be rendered as radiance fields, resulting in high-quality photo-realistic reconstructions of the non-rigidly deforming objects. We demonstrate that NPGs achieve superior results compared to previous works, especially in challenging scenarios with few multi-view cues.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.01196v2.pdf)

#### Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding

**Authors**: Shao-Hua Guan, Hao-Bin Duan, Miao Wang, Jin-Chuan Shi

<details>
<summary><b>Abstract</b></summary>
Open-vocabulary querying in 3D space is challenging but essential for scene understanding tasks such as object localization and segmentation. Language-embedded scene representations have made progress by incorporating language features into 3D spaces. However, their efficacy heavily depends on neural networks that are resource-intensive in training and rendering. Although recent 3D Gaussians offer efficient and high-quality novel view synthesis, directly embedding language features in them leads to prohibitive memory usage and decreased performance. In this work, we introduce Language Embedded 3D Gaussians, a novel scene representation for open-vocabulary query tasks. Instead of embedding high-dimensional raw semantic features on 3D Gaussians, we propose a dedicated quantization scheme that drastically alleviates the memory requirement, and a novel embedding procedure that achieves smoother yet high accuracy query, countering the multi-view feature inconsistencies and the high-frequency inductive bias in point-based representations. Our comprehensive experiments show that our representation achieves the best visual quality and language querying accuracy across current language-embedded representations, while maintaining real-time rendering frame rates on a single desktop GPU.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.18482v1.pdf)

#### MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes

**Authors**: Jeffrey Ichnowski, Shuran Song, Mike Zheng Shou, Jia-Wei Liu, Yunchao Yao, Zhao Mandi, Bardienus P. Duisterhof

<details>
<summary><b>Abstract</b></summary>
Accurate 3D tracking in highly deformable scenes with occlusions and shadows can facilitate new applications in robotics, augmented reality, and generative AI. However, tracking under these conditions is extremely challenging due to the ambiguity that arises with large deformations, shadows, and occlusions. We introduce MD-Splatting, an approach for simultaneous 3D tracking and novel view synthesis, using video captures of a dynamic scene from various camera poses. MD-Splatting builds on recent advances in Gaussian splatting, a method that learns the properties of a large number of Gaussians for state-of-the-art and fast novel view synthesis. MD-Splatting learns a deformation function to project a set of Gaussians with non-metric, thus canonical, properties into metric space. The deformation function uses a neural-voxel encoding and a multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow scalar. We enforce physics-inspired regularization terms based on local rigidity, conservation of momentum, and isometry, which leads to trajectories with smaller trajectory errors. MD-Splatting achieves high-quality 3D tracking on highly deformable scenes with shadows and occlusions. Compared to state-of-the-art, we improve 3D tracking by an average of 23.9 %, while simultaneously achieving high-quality novel view synthesis. With sufficient texture such as in scene 6, MD-Splatting achieves a median tracking error of 3.39 mm on a cloth of 1 x 1 meters in size. Project website: https://md-splatting.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00583v1.pdf)

#### GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering

**Authors**: Guocheng Qian, Jinjie Mai, Andrea Vedaldi, Bernard Ghanem, Carl Vondrick, Ruoshi Liu, Luke Melas-Kyriazi, Abdullah Hamdi

<details>
<summary><b>Abstract</b></summary>
Advancements in 3D Gaussian Splatting have significantly accelerated 3D reconstruction and generation. However, it may require a large number of Gaussians, which creates a substantial memory footprint. This paper introduces GES (Generalized Exponential Splatting), a novel representation that employs Generalized Exponential Function (GEF) to model 3D scenes, requiring far fewer particles to represent a scene and thus significantly outperforming Gaussian Splatting methods in efficiency with a plug-and-play replacement ability for Gaussian-based utilities. GES is validated theoretically and empirically in both principled 1D setup and realistic 3D scenes. It is shown to represent signals with sharp edges more accurately, which are typically challenging for Gaussians due to their inherent low-pass characteristics. Our empirical analysis demonstrates that GEF outperforms Gaussians in fitting natural-occurring signals (e.g. squares, triangles, and parabolic signals), thereby reducing the need for extensive splitting operations that increase the memory footprint of Gaussian Splatting. With the aid of a frequency-modulated loss, GES achieves competitive performance in novel-view synthesis benchmarks while requiring less than half the memory storage of Gaussian Splatting and increasing the rendering speed by up to 39%. The code is available on the project website https://abdullahamdi.com/ges .
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10128v2.pdf)

#### CoGS: Controllable Gaussian Splatting

**Authors**: László A. Jeni, Koichiro Niinuma, Zoltán Á. Milacski, Joel Julin, Heng Yu

<details>
<summary><b>Abstract</b></summary>
Capturing and re-animating the 3D structure of articulated objects present significant barriers. On one hand, methods requiring extensively calibrated multi-view setups are prohibitively complex and resource-intensive, limiting their practical applicability. On the other hand, while single-camera Neural Radiance Fields (NeRFs) offer a more streamlined approach, they have excessive training and rendering costs. 3D Gaussian Splatting would be a suitable alternative but for two reasons. Firstly, existing methods for 3D dynamic Gaussians require synchronized multi-view cameras, and secondly, the lack of controllability in dynamic scenarios. We present CoGS, a method for Controllable Gaussian Splatting, that enables the direct manipulation of scene elements, offering real-time control of dynamic scenes without the prerequisite of pre-computing control signals. We evaluated CoGS using both synthetic and real-world datasets that include dynamic objects that differ in degree of difficulty. In our evaluations, CoGS consistently outperformed existing dynamic and controllable neural representations in terms of visual fidelity.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.05664v2.pdf)

#### Compact 3D Gaussian Representation for Radiance Field

**Authors**: Eunbyung Park, Jong Hwan Ko, Xiangyu Sun, Daniel Rho, Joo Chan Lee

<details>
<summary><b>Abstract</b></summary>
Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussisan-based representation and adopts the rasterization pipeline to render the images rather than volumetric rendering, achieving very fast rendering speed and promising image quality. However, a significant drawback arises as 3DGS entails a substantial number of 3D Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric attributes of Gaussian by vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25$\times$ reduced storage and enhanced rendering speed, while maintaining the quality of the scene representation, compared to 3DGS. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13681v2.pdf)

#### 4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping for Monocular, Stereo, and RGB-D Cameras

**Authors**: Sai-Kit Yeung, Hui Cheng, Longwei Li, Huajian Huang

<details>
<summary><b>Abstract</b></summary>
The integration of neural rendering and the SLAM system recently showed promising results in joint localization and photorealistic view reconstruction. However, existing methods, fully relying on implicit representations, are so resource-hungry that they cannot run on portable devices, which deviates from the original intention of SLAM. In this paper, we present Photo-SLAM, a novel SLAM framework with a hyper primitives map. Specifically, we simultaneously exploit explicit geometric features for localization and learn implicit photometric features to represent the texture information of the observed environment. In addition to actively densifying hyper primitives based on geometric features, we further introduce a Gaussian-Pyramid-based training method to progressively learn multi-level features, enhancing photorealistic mapping performance. The extensive experiments with monocular, stereo, and RGB-D datasets prove that our proposed system Photo-SLAM significantly outperforms current state-of-the-art SLAM systems for online photorealistic mapping, e.g., PSNR is 30% higher and rendering speed is hundreds of times faster in the Replica dataset. Moreover, the Photo-SLAM can run at real-time speed using an embedded platform such as Jetson AGX Orin, showing the potential of robotics applications.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16728v2.pdf)

#### GART: Gaussian Articulated Template Models

**Authors**: Kostas Daniilidis, Lingjie Liu, Georgios Pavlakos, Yufu Wang, Jiahui Lei

<details>
<summary><b>Abstract</b></summary>
We introduce Gaussian Articulated Template Model GART, an explicit, efficient, and expressive representation for non-rigid articulated subject capturing and rendering from monocular videos. GART utilizes a mixture of moving 3D Gaussians to explicitly approximate a deformable subject's geometry and appearance. It takes advantage of a categorical template model prior (SMPL, SMAL, etc.) with learnable forward skinning while further generalizing to more complex non-rigid deformations with novel latent bones. GART can be reconstructed via differentiable rendering from monocular videos in seconds or minutes and rendered in novel poses faster than 150fps.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16099v1.pdf)

#### Gaussian Grouping: Segment and Edit Anything in 3D Scenes

**Authors**: Lei Ke, Fisher Yu, Martin Danelljan, Mingqiao Ye

<details>
<summary><b>Abstract</b></summary>
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by Segment Anything Model (SAM), along with introduced 3D spatial consistency regularization. Compared to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization, style transfer and scene recomposition. Our code and models are at https://github.com/lkeab/gaussian-grouping.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00732v2.pdf)

#### Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis

**Authors**: Rüdiger Westermann, Josef Stumpfegger, Simon Niedermayr

<details>
<summary><b>Abstract</b></summary>
Recently, high-fidelity scene reconstruction with an optimized 3D Gaussian splat representation has been introduced for novel view synthesis from sparse image sets. Making such representations suitable for applications like network streaming and rendering on low-power devices requires significantly reduced memory consumption as well as improved rendering efficiency. We propose a compressed 3D Gaussian splat representation that utilizes sensitivity-aware vector clustering with quantization-aware training to compress directional colors and Gaussian parameters. The learned codebooks have low bitrates and achieve a compression rate of up to $31\times$ on real-world scenes with only minimal degradation of visual quality. We demonstrate that the compressed splat representation can be efficiently rendered with hardware rasterization on lightweight GPUs at up to $4\times$ higher framerates than reported via an optimized GPU compute pipeline. Extensive experiments across multiple datasets demonstrate the robustness and rendering speed of the proposed approach.
</details>

[📄 Paper](https://arxiv.org/pdf/2401.02436v2.pdf)

#### pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction

**Authors**: Vincent Sitzmann, Andrea Tagliasacchi, Sizhe Li, David Charatan

<details>
<summary><b>Abstract</b></summary>
We introduce pixelSplat, a feed-forward model that learns to reconstruct 3D radiance fields parameterized by 3D Gaussian primitives from pairs of images. Our model features real-time and memory-efficient rendering for scalable training as well as fast 3D reconstruction at inference time. To overcome local minima inherent to sparse and locally supported representations, we predict a dense probability distribution over 3D and sample Gaussian means from that probability distribution. We make this sampling operation differentiable via a reparameterization trick, allowing us to back-propagate gradients through the Gaussian splatting representation. We benchmark our method on wide-baseline novel view synthesis on the real-world RealEstate10k and ACID datasets, where we outperform state-of-the-art light field transformers and accelerate rendering by 2.5 orders of magnitude while reconstructing an interpretable and editable 3D radiance field.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.12337v4.pdf)

#### GaussianDiffusion: 3D Gaussian Splatting for Denoising Diffusion Probabilistic Models with Structured Noise

**Authors**: Kuo-Kun Tseng, Huaibin Wang, Xinhai Li

<details>
<summary><b>Abstract</b></summary>
Text-to-3D, known for its efficient generation methods and expansive creative potential, has garnered significant attention in the AIGC domain. However, the amalgamation of Nerf and 2D diffusion models frequently yields oversaturated images, posing severe limitations on downstream industrial applications due to the constraints of pixelwise rendering method. Gaussian splatting has recently superseded the traditional pointwise sampling technique prevalent in NeRF-based methodologies, revolutionizing various aspects of 3D reconstruction. This paper introduces a novel text to 3D content generation framework based on Gaussian splatting, enabling fine control over image saturation through individual Gaussian sphere transparencies, thereby producing more realistic images. The challenge of achieving multi-view consistency in 3D generation significantly impedes modeling complexity and accuracy. Taking inspiration from SJC, we explore employing multi-view noise distributions to perturb images generated by 3D Gaussian splatting, aiming to rectify inconsistencies in multi-view geometry. We ingeniously devise an efficient method to generate noise that produces Gaussian noise from diverse viewpoints, all originating from a shared noise source. Furthermore, vanilla 3D Gaussian-based generation tends to trap models in local minima, causing artifacts like floaters, burrs, or proliferative elements. To mitigate these issues, we propose the variational Gaussian splatting technique to enhance the quality and stability of 3D appearance. To our knowledge, our approach represents the first comprehensive utilization of Gaussian splatting across the entire spectrum of 3D content generation processes.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11221v1.pdf)

#### DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting

**Authors**: Kostas Daniilidis, Jiahui Lei, Agelos Kratimenos

<details>
<summary><b>Abstract</b></summary>
Accurately and efficiently modeling dynamic scenes and motions is considered so challenging a task due to temporal dynamics and motion complexity. To address these challenges, we propose DynMF, a compact and efficient representation that decomposes a dynamic scene into a few neural trajectories. We argue that the per-point motions of a dynamic scene can be decomposed into a small set of explicit or learned trajectories. Our carefully designed neural framework consisting of a tiny set of learned basis queried only in time allows for rendering speed similar to 3D Gaussian Splatting, surpassing 120 FPS, while at the same time, requiring only double the storage compared to static scenes. Our neural representation adequately constrains the inherently underconstrained motion field of a dynamic scene leading to effective and fast optimization. This is done by biding each point to motion coefficients that enforce the per-point sharing of basis trajectories. By carefully applying a sparsity loss to the motion coefficients, we are able to disentangle the motions that comprise the scene, independently control them, and generate novel motion combinations that have never been seen before. We can reach state-of-the-art render quality within just 5 minutes of training and in less than half an hour, we can synthesize novel views of dynamic scenes with superior photorealistic quality. Our representation is interpretable, efficient, and expressive enough to offer real-time view synthesis of complex dynamic scene motions, in monocular and multi-view scenarios.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00112v1.pdf)

#### Splatter Image: Ultra-Fast Single-View 3D Reconstruction

**Authors**: Andrea Vedaldi, Christian Rupprecht, Stanislaw Szymanowicz

<details>
<summary><b>Abstract</b></summary>
We introduce the \method, an ultra-efficient approach for monocular 3D object reconstruction. Splatter Image is based on Gaussian Splatting, which allows fast and high-quality reconstruction of 3D scenes from multiple images. We apply Gaussian Splatting to monocular reconstruction by learning a neural network that, at test time, performs reconstruction in a feed-forward manner, at 38 FPS. Our main innovation is the surprisingly straightforward design of this network, which, using 2D operators, maps the input image to one 3D Gaussian per pixel. The resulting set of Gaussians thus has the form an image, the Splatter Image. We further extend the method take several images as input via cross-view attention. Owning to the speed of the renderer (588 FPS), we use a single GPU for training while generating entire images at each iteration to optimize perceptual metrics like LPIPS. On several synthetic, real, multi-category and large-scale benchmark datasets, we achieve better results in terms of PSNR, LPIPS, and other metrics while training and evaluating much faster than prior works. Code, models, demo and more results are available at https://szymanowiczs.github.io/splatter-image.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13150v2.pdf)

#### Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction

**Authors**: Houqiang Li, Min Wang, Li Li, Wengang Zhou, Zhiyang Guo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has become an emerging tool for dynamic scene reconstruction. However, existing methods focus mainly on extending static 3DGS into a time-variant representation, while overlooking the rich motion information carried by 2D observations, thus suffering from performance degradation and model redundancy. To address the above problem, we propose a novel motion-aware enhancement framework for dynamic scene reconstruction, which mines useful motion cues from optical flow to improve different paradigms of dynamic 3DGS. Specifically, we first establish a correspondence between 3D Gaussian movements and pixel-level flow. Then a novel flow augmentation method is introduced with additional insights into uncertainty and loss collaboration. Moreover, for the prevalent deformation-based paradigm that presents a harder optimization problem, a transient-aware deformation auxiliary module is proposed. We conduct extensive experiments on both multi-view and monocular scenes to verify the merits of our work. Compared with the baselines, our method shows significant superiority in both rendering quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11447v1.pdf)

#### Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering

**Authors**: Gim Hee Lee, Yu Chen, Weng Fei Low, Zhiwen Yan

<details>
<summary><b>Abstract</b></summary>
3D Gaussians have recently emerged as a highly efficient representation for 3D reconstruction and rendering. Despite its high rendering quality and speed at high resolutions, they both deteriorate drastically when rendered at lower resolutions or from far away camera position. During low resolution or far away rendering, the pixel size of the image can fall below the Nyquist frequency compared to the screen size of each splatted 3D Gaussian and leads to aliasing effect. The rendering is also drastically slowed down by the sequential alpha blending of more splatted Gaussians per pixel. To address these issues, we propose a multi-scale 3D Gaussian splatting algorithm, which maintains Gaussians at different scales to represent the same scene. Higher-resolution images are rendered with more small Gaussians, and lower-resolution images are rendered with fewer larger Gaussians. With similar training time, our algorithm can achieve 13\%-66\% PSNR and 160\%-2400\% rendering speed improvement at 4$\times$-128$\times$ scale rendering on Mip-NeRF360 dataset compared to the single scale 3D Gaussian splitting. Our code and more results are available on our project website https://jokeryan.github.io/projects/ms-gs/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17089v2.pdf)

#### Point'n Move: Interactive Scene Object Manipulation on Gaussian Splatting Radiance Fields

**Authors**: Hongchuan Yu, Jiajun Huang

<details>
<summary><b>Abstract</b></summary>
We propose Point'n Move, a method that achieves interactive scene object manipulation with exposed region inpainting. Interactivity here further comes from intuitive object selection and real-time editing. To achieve this, we adopt Gaussian Splatting Radiance Field as the scene representation and fully leverage its explicit nature and speed advantage. Its explicit representation formulation allows us to devise a 2D prompt points to 3D mask dual-stage self-prompting segmentation algorithm, perform mask refinement and merging, minimize change as well as provide good initialization for scene inpainting and perform editing in real-time without per-editing training, all leads to superior quality and performance. We test our method by performing editing on both forward-facing and 360 scenes. We also compare our method against existing scene object removal methods, showing superior quality despite being more capable and having a speed advantage.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.16737v1.pdf)

## Splatting
#### ADOP: Approximate Differentiable One-Pixel Point Rendering

**Authors**: Marc Stamminger, Linus Franke, Darius Rückert

<details>
<summary><b>Abstract</b></summary>
In this paper we present ADOP, a novel point-based, differentiable neural rendering pipeline. Like other neural renderers, our system takes as input calibrated camera images and a proxy geometry of the scene, in our case a point cloud. To generate a novel view, the point cloud is rasterized with learned feature vectors as colors and a deep neural network fills the remaining holes and shades each output pixel. The rasterizer renders points as one-pixel splats, which makes it very fast and allows us to compute gradients with respect to all relevant input parameters efficiently. Furthermore, our pipeline contains a fully differentiable physically-based photometric camera model, including exposure, white balance, and a camera response function. Following the idea of inverse rendering, we use our renderer to refine its input in order to reduce inconsistencies and optimize the quality of its output. In particular, we can optimize structural parameters like the camera pose, lens distortions, point positions and features, and a neural environment map, but also photometric parameters like camera response function, vignetting, and per-image exposure and white balance. Because our pipeline includes photometric parameters, e.g.~exposure and camera response function, our system can smoothly handle input images with varying exposure and white balance, and generates high-dynamic range output. We show that due to the improved input, we can achieve high render quality, also for difficult input, e.g. with imperfect camera calibrations, inaccurate proxy geometry, or varying exposure. As a result, a simpler and thus faster deep neural network is sufficient for reconstruction. In combination with the fast point rasterization, ADOP achieves real-time rendering rates even for models with well over 100M points. https://github.com/darglein/ADOP
</details>

[📄 Paper](https://arxiv.org/pdf/2110.06635v3.pdf)

#### GS++: Error Analyzing and Optimal Gaussian Splatting
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering

**Authors**: Marc Stamminger, Laura Fink, Darius Rückert, Linus Franke

<details>
<summary><b>Abstract</b></summary>
Point-based radiance field rendering has demonstrated impressive results for novel view synthesis, offering a compelling blend of rendering quality and computational efficiency. However, also latest approaches in this domain are not without their shortcomings. 3D Gaussian Splatting [Kerbl and Kopanas et al. 2023] struggles when tasked with rendering highly detailed scenes, due to blurring and cloudy artifacts. On the other hand, ADOP [R\"uckert et al. 2022] can accommodate crisper images, but the neural reconstruction network decreases performance, it grapples with temporal instability and it is unable to effectively address large gaps in the point cloud. In this paper, we present TRIPS (Trilinear Point Splatting), an approach that combines ideas from both Gaussian Splatting and ADOP. The fundamental concept behind our novel technique involves rasterizing points into a screen-space image pyramid, with the selection of the pyramid layer determined by the projected point size. This approach allows rendering arbitrarily large points using a single trilinear write. A lightweight neural network is then used to reconstruct a hole-free image including detail beyond splat resolution. Importantly, our render pipeline is entirely differentiable, allowing for automatic optimization of both point sizes and positions. Our evaluation demonstrate that TRIPS surpasses existing state-of-the-art methods in terms of rendering quality while maintaining a real-time frame rate of 60 frames per second on readily available hardware. This performance extends to challenging scenarios, such as scenes featuring intricate geometry, expansive landscapes, and auto-exposed footage. The project page is located at: https://lfranke.github.io/trips/
</details>

[📄 Paper](https://arxiv.org/pdf/2401.06003v2.pdf)

#### 3D Gaussian Splatting for Real-Time Radiance Field Rendering

**Authors**: George Drettakis, Thomas Leimkühler, Georgios Kopanas, Bernhard Kerbl

<details>
<summary><b>Abstract</b></summary>
Radiance Field methods have recently revolutionized novel-view synthesis of scenes captured with multiple photos or videos. However, achieving high visual quality still requires neural networks that are costly to train and render, while recent faster methods inevitably trade off speed for quality. For unbounded and complete scenes (rather than isolated objects) and 1080p resolution rendering, no current method can achieve real-time display rates. We introduce three key elements that allow us to achieve state-of-the-art visual quality while maintaining competitive training times and importantly allow high-quality real-time (>= 30 fps) novel-view synthesis at 1080p resolution. First, starting from sparse points produced during camera calibration, we represent the scene with 3D Gaussians that preserve desirable properties of continuous volumetric radiance fields for scene optimization while avoiding unnecessary computation in empty space; Second, we perform interleaved optimization/density control of the 3D Gaussians, notably optimizing anisotropic covariance to achieve an accurate representation of the scene; Third, we develop a fast visibility-aware rendering algorithm that supports anisotropic splatting and both accelerates training and allows realtime rendering. We demonstrate state-of-the-art visual quality and real-time rendering on several established datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2308.04079v1.pdf)

## Regularization
### 3D Regularization
#### CG3D: Compositional Generation for Text-to-3D via Gaussian Splatting

**Authors**: Achuta Kadambi, Pradyumna Chari, Alexander Vilesov

<details>
<summary><b>Abstract</b></summary>
With the onset of diffusion-based generative models and their ability to generate text-conditioned images, content generation has received a massive invigoration. Recently, these models have been shown to provide useful guidance for the generation of 3D graphics assets. However, existing work in text-conditioned 3D generation faces fundamental constraints: (i) inability to generate detailed, multi-object scenes, (ii) inability to textually control multi-object configurations, and (iii) physically realistic scene composition. In this work, we propose CG3D, a method for compositionally generating scalable 3D assets that resolves these constraints. We find that explicit Gaussian radiance fields, parameterized to allow for compositions of objects, possess the capability to enable semantically and physically consistent scenes. By utilizing a guidance framework built around this explicit representation, we show state of the art results, capable of even exceeding the guiding diffusion model in terms of object combinations and physics accuracy.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17907v1.pdf)

#### DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization

**Authors**: Lin Gu, Jun Zhou, Xin Ning, Jin Zheng, Xiao Bai, Jiawei Zhang, Jiahe Li

<details>
<summary><b>Abstract</b></summary>
Radiance fields have demonstrated impressive performance in synthesizing novel views from sparse input views, yet prevailing methods suffer from high training costs and slow inference speed. This paper introduces DNGaussian, a depth-regularized framework based on 3D Gaussian radiance fields, offering real-time and high-quality few-shot novel view synthesis at low costs. Our motivation stems from the highly efficient representation and surprising quality of the recent 3D Gaussian Splatting, despite it will encounter a geometry degradation when input views decrease. In the Gaussian radiance fields, we find this degradation in scene geometry primarily lined to the positioning of Gaussian primitives and can be mitigated by depth constraint. Consequently, we propose a Hard and Soft Depth Regularization to restore accurate scene geometry under coarse monocular depth supervision while maintaining a fine-grained color appearance. To further refine detailed geometry reshaping, we introduce Global-Local Depth Normalization, enhancing the focus on small local depth changes. Extensive experiments on LLFF, DTU, and Blender datasets demonstrate that DNGaussian outperforms state-of-the-art methods, achieving comparable or better results with significantly reduced memory cost, a $25 \times$ reduction in training time, and over $3000 \times$ faster rendering speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.06912v3.pdf)

#### MD-Splatting: Learning Metric Deformation from 4D Gaussians in Highly Deformable Scenes

**Authors**: Jeffrey Ichnowski, Shuran Song, Mike Zheng Shou, Jia-Wei Liu, Yunchao Yao, Zhao Mandi, Bardienus P. Duisterhof

<details>
<summary><b>Abstract</b></summary>
Accurate 3D tracking in highly deformable scenes with occlusions and shadows can facilitate new applications in robotics, augmented reality, and generative AI. However, tracking under these conditions is extremely challenging due to the ambiguity that arises with large deformations, shadows, and occlusions. We introduce MD-Splatting, an approach for simultaneous 3D tracking and novel view synthesis, using video captures of a dynamic scene from various camera poses. MD-Splatting builds on recent advances in Gaussian splatting, a method that learns the properties of a large number of Gaussians for state-of-the-art and fast novel view synthesis. MD-Splatting learns a deformation function to project a set of Gaussians with non-metric, thus canonical, properties into metric space. The deformation function uses a neural-voxel encoding and a multilayer perceptron (MLP) to infer Gaussian position, rotation, and a shadow scalar. We enforce physics-inspired regularization terms based on local rigidity, conservation of momentum, and isometry, which leads to trajectories with smaller trajectory errors. MD-Splatting achieves high-quality 3D tracking on highly deformable scenes with shadows and occlusions. Compared to state-of-the-art, we improve 3D tracking by an average of 23.9 %, while simultaneously achieving high-quality novel view synthesis. With sufficient texture such as in scene 6, MD-Splatting achieves a median tracking error of 3.39 mm on a cloth of 1 x 1 meters in size. Project website: https://md-splatting.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00583v1.pdf)

#### Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images

**Authors**: Kyoung Mu Lee, Jeongtaek Oh, JaeYoung Chung

<details>
<summary><b>Abstract</b></summary>
In this paper, we present a method to optimize Gaussian splatting with a limited number of images while avoiding overfitting. Representing a 3D scene by combining numerous Gaussian splats has yielded outstanding visual quality. However, it tends to overfit the training views when only a small number of images are available. To address this issue, we introduce a dense depth map as a geometry guide to mitigate overfitting. We obtained the depth map using a pre-trained monocular depth estimation model and aligning the scale and offset using sparse COLMAP feature points. The adjusted depth aids in the color-based optimization of 3D Gaussian splatting, mitigating floating artifacts, and ensuring adherence to geometric constraints. We verify the proposed method on the NeRF-LLFF dataset with varying numbers of few images. Our approach demonstrates robust geometry compared to the original method that relies solely on images. Project page: robot0321.github.io/DepthRegGS
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13398v3.pdf)

#### GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering

**Authors**: Federico Tombari, Gim Hee Lee, Guangyao Zhai, Yan Di, Chenyu Lyu, Yanyan Li

<details>
<summary><b>Abstract</b></summary>
During the Gaussian Splatting optimization process, the scene's geometry can gradually deteriorate if its structure is not deliberately preserved, especially in non-textured regions such as walls, ceilings, and furniture surfaces. This degradation significantly affects the rendering quality of novel views that deviate significantly from the viewpoints in the training data. To mitigate this issue, we propose a novel approach called GeoGaussian. Based on the smoothly connected areas observed from point clouds, this method introduces a novel pipeline to initialize thin Gaussians aligned with the surfaces, where the characteristic can be transferred to new generations through a carefully designed densification strategy. Finally, the pipeline ensures that the scene's geometry and texture are maintained through constrained optimization processes with explicit geometry constraints. Benefiting from the proposed architecture, the generative ability of 3D Gaussians is enhanced, especially in structured regions. Our proposed pipeline achieves state-of-the-art performance in novel view synthesis and geometric reconstruction, as evaluated qualitatively and quantitatively on public datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11324v2.pdf)

#### BAGS: Building Animatable Gaussian Splatting from a Monocular Video with Diffusion Priors

**Authors**: Baoquan Chen, Libin Liu, Weiyu Li, Qingzhe Gao, Tingyang Zhang

<details>
<summary><b>Abstract</b></summary>
Animatable 3D reconstruction has significant applications across various fields, primarily relying on artists' handcraft creation. Recently, some studies have successfully constructed animatable 3D models from monocular videos. However, these approaches require sufficient view coverage of the object within the input video and typically necessitate significant time and computational costs for training and rendering. This limitation restricts the practical applications. In this work, we propose a method to build animatable 3D Gaussian Splatting from monocular video with diffusion priors. The 3D Gaussian representations significantly accelerate the training and rendering process, and the diffusion priors allow the method to learn 3D models with limited viewpoints. We also present the rigid regularization to enhance the utilization of the priors. We perform an extensive evaluation across various real-world videos, demonstrating its superior performance compared to the current state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11427v1.pdf)

#### NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance

**Authors**: Gim Hee Lee, Chen Li, Hanlin Chen

<details>
<summary><b>Abstract</b></summary>
Existing neural implicit surface reconstruction methods have achieved impressive performance in multi-view 3D reconstruction by leveraging explicit geometry priors such as depth maps or point clouds as regularization. However, the reconstruction results still lack fine details because of the over-smoothed depth map or sparse point cloud. In this work, we propose a neural implicit surface reconstruction pipeline with guidance from 3D Gaussian Splatting to recover highly detailed surfaces. The advantage of 3D Gaussian Splatting is that it can generate dense point clouds with detailed structure. Nonetheless, a naive adoption of 3D Gaussian Splatting can fail since the generated points are the centers of 3D Gaussians that do not necessarily lie on the surface. We thus introduce a scale regularizer to pull the centers close to the surface by enforcing the 3D Gaussians to be extremely thin. Moreover, we propose to refine the point cloud from 3D Gaussians Splatting with the normal priors from the surface predicted by neural implicit models instead of using a fixed set of points as guidance. Consequently, the quality of surface reconstruction improves from the guidance of the more accurate 3D Gaussian splatting. By jointly optimizing the 3D Gaussian Splatting and the neural implicit model, our approach benefits from both representations and generates complete surfaces with intricate details. Experiments on Tanks and Temples verify the effectiveness of our proposed method.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00846v1.pdf)

#### GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces

**Authors**: Yuexin Ma, Wenping Wang, Xiaoxiao Long, Xifeng Gao, YuAn Liu, Jiadong Tu, Yingwenqi Jiang

<details>
<summary><b>Abstract</b></summary>
The advent of neural 3D Gaussians has recently brought about a revolution in the field of neural rendering, facilitating the generation of high-quality renderings at real-time speeds. However, the explicit and discrete representation encounters challenges when applied to scenes featuring reflective surfaces. In this paper, we present GaussianShader, a novel method that applies a simplified shading function on 3D Gaussians to enhance the neural rendering in scenes with reflective surfaces while preserving the training and rendering efficiency. The main challenge in applying the shading function lies in the accurate normal estimation on discrete 3D Gaussians. Specifically, we proposed a novel normal estimation framework based on the shortest axis directions of 3D Gaussians with a delicately designed loss to make the consistency between the normals and the geometries of Gaussian spheres. Experiments show that GaussianShader strikes a commendable balance between efficiency and visual quality. Our method surpasses Gaussian Splatting in PSNR on specular object datasets, exhibiting an improvement of 1.57dB. When compared to prior works handling reflective surfaces, such as Ref-NeRF, our optimization time is significantly accelerated (23h vs. 0.58h). Please click on our project website to see more results.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17977v1.pdf)

#### Gaussian Grouping: Segment and Edit Anything in 3D Scenes

**Authors**: Lei Ke, Fisher Yu, Martin Danelljan, Mingqiao Ye

<details>
<summary><b>Abstract</b></summary>
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by Segment Anything Model (SAM), along with introduced 3D spatial consistency regularization. Compared to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization, style transfer and scene recomposition. Our code and models are at https://github.com/lkeab/gaussian-grouping.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00732v2.pdf)

#### Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle

**Authors**: Yao Yao, Siyu Zhu, Zuozhuo Dai, Youtian Lin

<details>
<summary><b>Abstract</b></summary>
We introduce Gaussian-Flow, a novel point-based approach for fast dynamic scene reconstruction and real-time rendering from both multi-view and monocular videos. In contrast to the prevalent NeRF-based approaches hampered by slow training and rendering speeds, our approach harnesses recent advancements in point-based 3D Gaussian Splatting (3DGS). Specifically, a novel Dual-Domain Deformation Model (DDDM) is proposed to explicitly model attribute deformations of each Gaussian point, where the time-dependent residual of each attribute is captured by a polynomial fitting in the time domain, and a Fourier series fitting in the frequency domain. The proposed DDDM is capable of modeling complex scene deformations across long video footage, eliminating the need for training separate 3DGS for each frame or introducing an additional implicit neural field to model 3D dynamics. Moreover, the explicit deformation modeling for discretized Gaussian points ensures ultra-fast training and rendering of a 4D scene, which is comparable to the original 3DGS designed for static 3D reconstruction. Our proposed approach showcases a substantial efficiency improvement, achieving a $5\times$ faster training speed compared to the per-frame 3DGS modeling. In addition, quantitative results demonstrate that the proposed Gaussian-Flow significantly outperforms previous leading methods in novel view rendering quality. Project page: https://nju-3dv.github.io/projects/Gaussian-Flow
</details>
[📄 Paper](https://arxiv.org/pdf/2312.03431v1.pdf)

#### High-quality Surface Reconstruction using Gaussian Surfels

**Authors**: Weiwei Xu, Huamin Wang, Xinguo Liu, Wenxiang Xie, Jiamin Xu, Pinxuan Dai

<details>
<summary><b>Abstract</b></summary>
We propose a novel point-based representation, Gaussian surfels, to combine the advantages of the flexible optimization procedure in 3D Gaussian points and the surface alignment property of surfels. This is achieved by directly setting the z-scale of 3D Gaussian points to 0, effectively flattening the original 3D ellipsoid into a 2D ellipse. Such a design provides clear guidance to the optimizer. By treating the local z-axis as the normal direction, it greatly improves optimization stability and surface alignment. While the derivatives to the local z-axis computed from the covariance matrix are zero in this setting, we design a self-supervised normal-depth consistency loss to remedy this issue. Monocular normal priors and foreground masks are incorporated to enhance the quality of the reconstruction, mitigating issues related to highlights and background. We propose a volumetric cutting method to aggregate the information of Gaussian surfels so as to remove erroneous points in depth maps generated by alpha blending. Finally, we apply screened Poisson reconstruction method to the fused depth maps to extract the surface mesh. Experimental results show that our method demonstrates superior performance in surface reconstruction compared to state-of-the-art neural volume rendering and point-based rendering methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.17774v2.pdf)

### 2D Regularization
#### High-Fidelity SLAM Using Gaussian Splatting with Rendering-Guided Densification and Regularized Optimization

**Authors**: Martin Magnusson, Achim J. Lilienthal, Malcolm Mielle, Shuo Sun

<details>
<summary><b>Abstract</b></summary>
We propose a dense RGBD SLAM system based on 3D Gaussian Splatting that provides metrically accurate pose tracking and visually realistic reconstruction. To this end, we first propose a Gaussian densification strategy based on the rendering loss to map unobserved areas and refine reobserved areas. Second, we introduce extra regularization parameters to alleviate the forgetting problem in the continuous mapping problem, where parameters tend to overfit the latest frame and result in decreasing rendering quality for previous frames. Both mapping and tracking are performed with Gaussian parameters by minimizing re-rendering loss in a differentiable way. Compared to recent neural and concurrently developed gaussian splatting RGBD SLAM baselines, our method achieves state-of-the-art results on the synthetic dataset Replica and competitive results on the real-world dataset TUM.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.12535v1.pdf)

#### Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond

**Authors**: Mingyuan Zhou, Amir Sadeghian, Ali Sadeghian, Huangjie Zheng, Mohammadreza Armandpour

<details>
<summary><b>Abstract</b></summary>
Although text-to-image diffusion models have made significant strides in generating images from text, they are sometimes more inclined to generate images like the data on which the model was trained rather than the provided text. This limitation has hindered their usage in both 2D and 3D applications. To address this problem, we explored the use of negative prompts but found that the current implementation fails to produce desired results, particularly when there is an overlap between the main and negative prompts. To overcome this issue, we propose Perp-Neg, a new algorithm that leverages the geometrical properties of the score space to address the shortcomings of the current negative prompts algorithm. Perp-Neg does not require any training or fine-tuning of the model. Moreover, we experimentally demonstrate that Perp-Neg provides greater flexibility in generating images by enabling users to edit out unwanted concepts from the initially generated images in 2D cases. Furthermore, to extend the application of Perp-Neg to 3D, we conducted a thorough exploration of how Perp-Neg can be used in 2D to condition the diffusion model to generate desired views, rather than being biased toward the canonical views. Finally, we applied our 2D intuition to integrate Perp-Neg with the state-of-the-art text-to-3D (DreamFusion) method, effectively addressing its Janus (multi-head) problem. Our project page is available at https://Perp-Neg.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2304.04968v3.pdf)

#### FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

**Authors**: Zhangyang Wang, Yifan Jiang, Zhiwen Fan, Zehao Zhu

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender. Project website: https://zehaozhu.github.io/FSGS/.
</details>
[📄 Paper](https://arxiv.org/pdf/2312.00451v2.pdf)

#### Learn to Optimize Denoising Scores for 3D Generation: A Unified and Improved Diffusion Prior on NeRF and 3D Gaussian Splatting

**Authors**: Guosheng Lin, Fayao Liu, Xulei Yang, Yi Xu, Chi Zhang, Cheng Chen, YiWen Chen, Xiaofeng Yang

<details>
<summary><b>Abstract</b></summary>
We propose a unified framework aimed at enhancing the diffusion priors for 3D generation tasks. Despite the critical importance of these tasks, existing methodologies often struggle to generate high-caliber results. We begin by examining the inherent limitations in previous diffusion priors. We identify a divergence between the diffusion priors and the training procedures of diffusion models that substantially impairs the quality of 3D generation. To address this issue, we propose a novel, unified framework that iteratively optimizes both the 3D model and the diffusion prior. Leveraging the different learnable parameters of the diffusion prior, our approach offers multiple configurations, affording various trade-offs between performance and implementation complexity. Notably, our experimental results demonstrate that our method markedly surpasses existing techniques, establishing new state-of-the-art in the realm of text-to-3D generation. Furthermore, our approach exhibits impressive performance on both NeRF and the newly introduced 3D Gaussian Splatting backbones. Additionally, our framework yields insightful contributions to the understanding of recent score distillation methods, such as the VSD and DDS loss.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.04820v1.pdf)

#### Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models

**Authors**: Karsten Kreis, Sanja Fidler, Antonio Torralba, Seung Wook Kim, Huan Ling

<details>
<summary><b>Abstract</b></summary>
Text-guided diffusion models have revolutionized image and video generation and have also been successfully used for optimization-based 3D object synthesis. Here, we instead focus on the underexplored text-to-4D setting and synthesize dynamic, animated 3D objects using score distillation methods with an additional temporal dimension. Compared to previous work, we pursue a novel compositional generation-based approach, and combine text-to-image, text-to-video, and 3D-aware multiview diffusion models to provide feedback during 4D object optimization, thereby simultaneously enforcing temporal consistency, high-quality visual appearance and realistic geometry. Our method, called Align Your Gaussians (AYG), leverages dynamic 3D Gaussian Splatting with deformation fields as 4D representation. Crucial to AYG is a novel method to regularize the distribution of the moving 3D Gaussians and thereby stabilize the optimization and induce motion. We also propose a motion amplification mechanism as well as a new autoregressive synthesis scheme to generate and combine multiple 4D sequences for longer generation. These techniques allow us to synthesize vivid dynamic scenes, outperform previous work qualitatively and quantitatively and achieve state-of-the-art text-to-4D performance. Due to the Gaussian 4D representation, different 4D animations can be seamlessly combined, as we demonstrate. AYG opens up promising avenues for animation, simulation and digital content creation as well as synthetic data generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13763v2.pdf)

#### FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization

**Authors**: Eric Xing, Shijian Lu, Muyu Xu, Fangneng Zhan, Jiahui Zhang

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting has achieved very impressive performance in real-time novel view synthesis. However, it often suffers from over-reconstruction during Gaussian densification where high-variance image regions are covered by a few large Gaussians only, leading to blur and artifacts in the rendered images. We design a progressive frequency regularization (FreGS) technique to tackle the over-reconstruction issue within the frequency space. Specifically, FreGS performs coarse-to-fine Gaussian densification by exploiting low-to-high frequency components that can be easily extracted with low-pass and high-pass filters in the Fourier space. By minimizing the discrepancy between the frequency spectrum of the rendered image and the corresponding ground truth, it achieves high-quality Gaussian densification and alleviates the over-reconstruction of Gaussian splatting effectively. Experiments over multiple widely adopted benchmarks (e.g., Mip-NeRF360, Tanks-and-Temples and Deep Blending) show that FreGS achieves superior novel view synthesis and outperforms the state-of-the-art consistently.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.06908v2.pdf)

#### HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

**Authors**: Ziwei Liu, Xihui Liu, Dahua Lin, Gang Zeng, Ying Shan, Jiaxiang Tang, Xiaohang Zhan, Xian Liu

<details>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17061v2.pdf)

#### Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting

**Authors**: Martin R. Oswald, Theo Gevers, Yue Li, Vladimir Yugay

<details>
<summary><b>Abstract</b></summary>
We present a dense simultaneous localization and mapping (SLAM) method that uses 3D Gaussians as a scene representation. Our approach enables interactive-time reconstruction and photo-realistic rendering from real-world single-camera RGBD videos. To this end, we propose a novel effective strategy for seeding new Gaussians for newly explored areas and their effective online optimization that is independent of the scene size and thus scalable to larger scenes. This is achieved by organizing the scene into sub-maps which are independently optimized and do not need to be kept in memory. We further accomplish frame-to-model camera tracking by minimizing photometric and geometric losses between the input and rendered frames. The Gaussian representation allows for high-quality photo-realistic real-time rendering of real-world scenes. Evaluation on synthetic and real-world datasets demonstrates competitive or superior performance in mapping, tracking, and rendering compared to existing neural dense SLAM methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.10070v2.pdf)

#### InstructPix2Pix: Learning to Follow Image Editing Instructions

**Authors**: Alexei A. Efros, Aleksander Holynski, Tim Brooks

<details>
<summary><b>Abstract</b></summary>
We propose a method for editing images from human instructions: given an input image and a written instruction that tells the model what to do, our model follows these instructions to edit the image. To obtain training data for this problem, we combine the knowledge of two large pretrained models -- a language model (GPT-3) and a text-to-image model (Stable Diffusion) -- to generate a large dataset of image editing examples. Our conditional diffusion model, InstructPix2Pix, is trained on our generated data, and generalizes to real images and user-written instructions at inference time. Since it performs edits in the forward pass and does not require per example fine-tuning or inversion, our model edits images quickly, in a matter of seconds. We show compelling editing results for a diverse collection of input images and written instructions.
</details>

[📄 Paper](https://arxiv.org/pdf/2211.09800v2.pdf)

#### Motion-aware 3D Gaussian Splatting for Efficient Dynamic Scene Reconstruction

**Authors**: Houqiang Li, Min Wang, Li Li, Wengang Zhou, Zhiyang Guo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has become an emerging tool for dynamic scene reconstruction. However, existing methods focus mainly on extending static 3DGS into a time-variant representation, while overlooking the rich motion information carried by 2D observations, thus suffering from performance degradation and model redundancy. To address the above problem, we propose a novel motion-aware enhancement framework for dynamic scene reconstruction, which mines useful motion cues from optical flow to improve different paradigms of dynamic 3DGS. Specifically, we first establish a correspondence between 3D Gaussian movements and pixel-level flow. Then a novel flow augmentation method is introduced with additional insights into uncertainty and loss collaboration. Moreover, for the prevalent deformation-based paradigm that presents a harder optimization problem, a transient-aware deformation auxiliary module is proposed. We conduct extensive experiments on both multi-view and monocular scenes to verify the merits of our work. Compared with the baselines, our method shows significant superiority in both rendering quality and efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11447v1.pdf)

#### GSEdit: Efficient Text-Guided Editing of 3D Objects via Gaussian Splatting

**Authors**: Emanuele Rodolà, Daniele Baieri, Andrea Sanchietti, Francesco Palandra

<details>
<summary><b>Abstract</b></summary>
We present GSEdit, a pipeline for text-guided 3D object editing based on Gaussian Splatting models. Our method enables the editing of the style and appearance of 3D objects without altering their main details, all in a matter of minutes on consumer hardware. We tackle the problem by leveraging Gaussian splatting to represent 3D scenes, and we optimize the model while progressively varying the image supervision by means of a pretrained image-based diffusion model. The input object may be given as a 3D triangular mesh, or directly provided as Gaussians from a generative model such as DreamGaussian. GSEdit ensures consistency across different viewpoints, maintaining the integrity of the original object's information. Compared to previously proposed methods relying on NeRF-like MLP models, GSEdit stands out for its efficiency, making 3D editing tasks much faster. Our editing process is refined via the application of the SDS loss, ensuring that our edits are both precise and accurate. Our comprehensive evaluation demonstrates that GSEdit effectively alters object shape and appearance following the given textual instructions while preserving their coherence and detail.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.05154v2.pdf)

#### LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching

**Authors**: Yingcong Chen, Xiaogang Xu, Haodong Li, Jiantao Lin, Xin Yang, Yixun Liang

<details>
<summary><b>Abstract</b></summary>
The recent advancements in text-to-3D generation mark a significant milestone in generative models, unlocking new possibilities for creating imaginative 3D assets across various real-world scenarios. While recent advancements in text-to-3D generation have shown promise, they often fall short in rendering detailed and high-quality 3D models. This problem is especially prevalent as many methods base themselves on Score Distillation Sampling (SDS). This paper identifies a notable deficiency in SDS, that it brings inconsistent and low-quality updating direction for the 3D model, causing the over-smoothing effect. To address this, we propose a novel approach called Interval Score Matching (ISM). ISM employs deterministic diffusing trajectories and utilizes interval-based score matching to counteract over-smoothing. Furthermore, we incorporate 3D Gaussian Splatting into our text-to-3D generation pipeline. Extensive experiments show that our model largely outperforms the state-of-the-art in quality and training efficiency.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11284v3.pdf)

## Training Strategy
### Multi-stage Training Strategy
#### LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

**Authors**: Zhangyang Wang, Dejia Xu, Zehao Zhu, Kairun Wen, Kevin Wang, Zhiwen Fan

<details>
<summary><b>Abstract</b></summary>
Recent advancements in real-time neural rendering using point-based techniques have paved the way for the widespread adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting come with a substantial storage overhead caused by growing the SfM points to millions, often demanding gigabyte-level disk space for a single unbounded scene, posing significant scalability challenges and hindering the splatting efficiency. To address this challenge, we introduce LightGaussian, a novel method designed to transform 3D Gaussians into a more efficient and compact format. Drawing inspiration from the concept of Network Pruning, LightGaussian identifies Gaussians that are insignificant in contributing to the scene reconstruction and adopts a pruning and recovery process, effectively reducing redundancy in Gaussian counts while preserving visual effects. Additionally, LightGaussian employs distillation and pseudo-view augmentation to distill spherical harmonics to a lower degree, allowing knowledge transfer to more compact representations while maintaining reflectance. Furthermore, we propose a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in lower bitwidth representations with minimal accuracy losses. In summary, LightGaussian achieves an averaged compression rate over 15x while boosting the FPS from 139 to 215, enabling an efficient representation of complex scenes on Mip-NeRF 360, Tank and Temple datasets. Project website: https://lightgaussian.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17245v5.pdf)

#### View-Consistent 3D Editing with Gaussian Splatting

**Authors**: Hanwang Zhang, Long Chen, Na Zhao, Zike Wu, Xuanyu Yi, Yuxuan Wang

<details>
<summary><b>Abstract</b></summary>
The advent of 3D Gaussian Splatting (3DGS) has revolutionized 3D editing, offering efficient, high-fidelity rendering and enabling precise local manipulations. Currently, diffusion-based 2D editing models are harnessed to modify multi-view rendered images, which then guide the editing of 3DGS models. However, this approach faces a critical issue of multi-view inconsistency, where the guidance images exhibit significant discrepancies across views, leading to mode collapse and visual artifacts of 3DGS. To this end, we introduce View-consistent Editing (VcEdit), a novel framework that seamlessly incorporates 3DGS into image editing processes, ensuring multi-view consistency in edited guidance images and effectively mitigating mode collapse issues. VcEdit employs two innovative consistency modules: the Cross-attention Consistency Module and the Editing Consistency Module, both designed to reduce inconsistencies in edited images. By incorporating these consistency modules into an iterative pattern, VcEdit proficiently resolves the issue of multi-view inconsistency, facilitating high-quality 3DGS editing across a diverse range of scenes. Further code and video results are released at http://yuxuanw.me/vcedit/.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11868v6.pdf)

#### Gaussian Splatting in Style

**Authors**: Daniel Cremers, Tarun Yenamandra, Cecilia Curreli, Mariia Gladkova, Abhishek Saroha

<details>
<summary><b>Abstract</b></summary>
Scene stylization extends the work of neural style transfer to three spatial dimensions. A vital challenge in this problem is to maintain the uniformity of the stylized appearance across a multi-view setting. A vast majority of the previous works achieve this by optimizing the scene with a specific style image. In contrast, we propose a novel architecture trained on a collection of style images, that at test time produces high quality stylized novel views. Our work builds up on the framework of 3D Gaussian splatting. For a given scene, we take the pretrained Gaussians and process them using a multi resolution hash grid and a tiny MLP to obtain the conditional stylised views. The explicit nature of 3D Gaussians give us inherent advantages over NeRF-based methods including geometric consistency, along with having a fast training and rendering regime. This enables our method to be useful for vast practical use cases such as in augmented or virtual reality applications. Through our experiments, we show our methods achieve state-of-the-art performance with superior visual quality on various indoor and outdoor real-world data.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.08498v1.pdf)

### End-to-End Training Strategy
#### LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

**Authors**: Zhangyang Wang, Dejia Xu, Zehao Zhu, Kairun Wen, Kevin Wang, Zhiwen Fan

<details>
<summary><b>Abstract</b></summary>
Recent advancements in real-time neural rendering using point-based techniques have paved the way for the widespread adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting come with a substantial storage overhead caused by growing the SfM points to millions, often demanding gigabyte-level disk space for a single unbounded scene, posing significant scalability challenges and hindering the splatting efficiency. To address this challenge, we introduce LightGaussian, a novel method designed to transform 3D Gaussians into a more efficient and compact format. Drawing inspiration from the concept of Network Pruning, LightGaussian identifies Gaussians that are insignificant in contributing to the scene reconstruction and adopts a pruning and recovery process, effectively reducing redundancy in Gaussian counts while preserving visual effects. Additionally, LightGaussian employs distillation and pseudo-view augmentation to distill spherical harmonics to a lower degree, allowing knowledge transfer to more compact representations while maintaining reflectance. Furthermore, we propose a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in lower bitwidth representations with minimal accuracy losses. In summary, LightGaussian achieves an averaged compression rate over 15x while boosting the FPS from 139 to 215, enabling an efficient representation of complex scenes on Mip-NeRF 360, Tank and Temple datasets. Project website: https://lightgaussian.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17245v5.pdf)

#### Hyper-3DG: Text-to-3D Gaussian Generation via Hypergraph

**Authors**: Yue Gao, Xun Yang, Wei Chen, Zhou Xue, Chaofan Luo, Jiahui Yang, Donglin Di

<details>
<summary><b>Abstract</b></summary>
Text-to-3D generation represents an exciting field that has seen rapid advancements, facilitating the transformation of textual descriptions into detailed 3D models. However, current progress often neglects the intricate high-order correlation of geometry and texture within 3D objects, leading to challenges such as over-smoothness, over-saturation and the Janus problem. In this work, we propose a method named ``3D Gaussian Generation via Hypergraph (Hyper-3DG)'', designed to capture the sophisticated high-order correlations present within 3D objects. Our framework is anchored by a well-established mainflow and an essential module, named ``Geometry and Texture Hypergraph Refiner (HGRefiner)''. This module not only refines the representation of 3D Gaussians but also accelerates the update process of these 3D Gaussians by conducting the Patch-3DGS Hypergraph Learning on both explicit attributes and latent visual features. Our framework allows for the production of finely generated 3D objects within a cohesive optimization, effectively circumventing degradation. Extensive experimentation has shown that our proposed method significantly enhances the quality of 3D generation while incurring no additional computational overhead for the underlying framework. (Project code: https://github.com/yjhboy/Hyper3DG)
</details>

[📄 Paper](https://arxiv.org/pdf/2403.09236v1.pdf)

#### GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

**Authors**: Guosheng Lin, Huaping Liu, Lei Yang, Zhongang Cai, Yikai Wang, Xiaofeng Yang, Feng Wang, Chi Zhang, Zilong Chen, YiWen Chen

<details>
<summary><b>Abstract</b></summary>
3D editing plays a crucial role in many areas such as gaming and virtual reality. Traditional 3D editing methods, which rely on representations like meshes and point clouds, often fall short in realistically depicting complex scenes. On the other hand, methods based on implicit 3D representations, like Neural Radiance Field (NeRF), render complex scenes effectively but suffer from slow processing speeds and limited control over specific scene areas. In response to these challenges, our paper presents GaussianEditor, an innovative and efficient 3D editing algorithm based on Gaussian Splatting (GS), a novel 3D representation. GaussianEditor enhances precision and control in editing through our proposed Gaussian semantic tracing, which traces the editing target throughout the training process. Additionally, we propose Hierarchical Gaussian splatting (HGS) to achieve stabilized and fine results under stochastic generative guidance from 2D diffusion models. We also develop editing strategies for efficient object removal and integration, a challenging task for existing methods. Our comprehensive experiments demonstrate GaussianEditor's superior control, efficacy, and rapid performance, marking a significant advancement in 3D editing. Project Page: https://buaacyw.github.io/gaussian-editor/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.14521v4.pdf)

#### GaussNav: Gaussian Splatting for Visual Navigation

**Authors**: Houqiang Li, Wengang Zhou, Min Wang, Xiaohan Lei

<details>
<summary><b>Abstract</b></summary>
In embodied vision, Instance ImageGoal Navigation (IIN) requires an agent to locate a specific object depicted in a goal image within an unexplored environment. The primary difficulty of IIN stems from the necessity of recognizing the target object across varying viewpoints and rejecting potential distractors. Existing map-based navigation methods largely adopt the representation form of Bird's Eye View (BEV) maps, which, however, lack the representation of detailed textures in a scene. To address the above issues, we propose a new Gaussian Splatting Navigation (abbreviated as GaussNav) framework for IIN task, which constructs a novel map representation based on 3D Gaussian Splatting (3DGS). The proposed framework enables the agent to not only memorize the geometry and semantic information of the scene, but also retain the textural features of objects. Our GaussNav framework demonstrates a significant leap in performance, evidenced by an increase in Success weighted by Path Length (SPL) from 0.252 to 0.578 on the challenging Habitat-Matterport 3D (HM3D) dataset. Our code will be made publicly available.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11625v2.pdf)

#### GaussianObject: Just Taking Four Images to Get A High-Quality 3D Object with Gaussian Splatting

**Authors**: Qi Tian, Wei Shen, Xiaopeng Zhang, Lingxi Xie, Ruofan Liang, Jiemin Fang, Sikuang Li, Chen Yang

<details>
<summary><b>Abstract</b></summary>
Reconstructing and rendering 3D objects from highly sparse views is of critical importance for promoting applications of 3D vision techniques and improving user experience. However, images from sparse views only contain very limited 3D information, leading to two significant challenges: 1) Difficulty in building multi-view consistency as images for matching are too few; 2) Partially omitted or highly compressed object information as view coverage is insufficient. To tackle these challenges, we propose GaussianObject, a framework to represent and render the 3D object with Gaussian splatting, that achieves high rendering quality with only 4 input images. We first introduce techniques of visual hull and floater elimination which explicitly inject structure priors into the initial optimization process for helping build multi-view consistency, yielding a coarse 3D Gaussian representation. Then we construct a Gaussian repair model based on diffusion models to supplement the omitted object information, where Gaussians are further refined. We design a self-generating strategy to obtain image pairs for training the repair model. Our GaussianObject is evaluated on several challenging datasets, including MipNeRF360, OmniObject3D, and OpenIllumination, achieving strong reconstruction results from only 4 views and significantly outperforming previous state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.10259v2.pdf)

#### Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting

**Authors**: Xiaogang Jin, Xiaojuan Qi, Shaohui Jiao, Wen Zhou, Xiaoyang Lyu, Yihua Huang, Yangtian Sun, Xinyu Gao, ZiYi Yang

<details>
<summary><b>Abstract</b></summary>
The recent advancements in 3D Gaussian splatting (3D-GS) have not only facilitated real-time rendering through modern GPU rasterization pipelines but have also attained state-of-the-art rendering quality. Nevertheless, despite its exceptional rendering quality and performance on standard datasets, 3D-GS frequently encounters difficulties in accurately modeling specular and anisotropic components. This issue stems from the limited ability of spherical harmonics (SH) to represent high-frequency information. To overcome this challenge, we introduce Spec-Gaussian, an approach that utilizes an anisotropic spherical Gaussian (ASG) appearance field instead of SH for modeling the view-dependent appearance of each 3D Gaussian. Additionally, we have developed a coarse-to-fine training strategy to improve learning efficiency and eliminate floaters caused by overfitting in real-world scenes. Our experimental results demonstrate that our method surpasses existing approaches in terms of rendering quality. Thanks to ASG, we have significantly improved the ability of 3D-GS to model scenes with specular and anisotropic components without increasing the number of 3D Gaussians. This improvement extends the applicability of 3D GS to handle intricate scenarios with specular and anisotropic surfaces.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.15870v1.pdf)

#### GeoGaussian: Geometry-aware Gaussian Splatting for Scene Rendering

**Authors**: Federico Tombari, Gim Hee Lee, Guangyao Zhai, Yan Di, Chenyu Lyu, Yanyan Li

<details>
<summary><b>Abstract</b></summary>
During the Gaussian Splatting optimization process, the scene's geometry can gradually deteriorate if its structure is not deliberately preserved, especially in non-textured regions such as walls, ceilings, and furniture surfaces. This degradation significantly affects the rendering quality of novel views that deviate significantly from the viewpoints in the training data. To mitigate this issue, we propose a novel approach called GeoGaussian. Based on the smoothly connected areas observed from point clouds, this method introduces a novel pipeline to initialize thin Gaussians aligned with the surfaces, where the characteristic can be transferred to new generations through a carefully designed densification strategy. Finally, the pipeline ensures that the scene's geometry and texture are maintained through constrained optimization processes with explicit geometry constraints. Benefiting from the proposed architecture, the generative ability of 3D Gaussians is enhanced, especially in structured regions. Our proposed pipeline achieves state-of-the-art performance in novel view synthesis and geometric reconstruction, as evaluated qualitatively and quantitatively on public datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11324v2.pdf)

#### 3DGS-Calib: 3D Gaussian Splatting for Multimodal SpatioTemporal Calibration

**Authors**: Cédric Demonceaux, Pascal Vasseur, Cyrille Migniot, Dzmitry Tsishkou, Luis Roldao, Nathan Piasco, Arthur Moreau, Moussab Bennehar, Quentin Herau

<details>
<summary><b>Abstract</b></summary>
Reliable multimodal sensor fusion algorithms re- quire accurate spatiotemporal calibration. Recently, targetless calibration techniques based on implicit neural representations have proven to provide precise and robust results. Nevertheless, such methods are inherently slow to train given the high compu- tational overhead caused by the large number of sampled points required for volume rendering. With the recent introduction of 3D Gaussian Splatting as a faster alternative to implicit representation methods, we propose to leverage this new ren- dering approach to achieve faster multi-sensor calibration. We introduce 3DGS-Calib, a new calibration method that relies on the speed and rendering accuracy of 3D Gaussian Splatting to achieve multimodal spatiotemporal calibration that is accurate, robust, and with a substantial speed-up compared to methods relying on implicit neural representations. We demonstrate the superiority of our proposal with experimental results on sequences from KITTI-360, a widely used driving dataset.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11577v1.pdf)

#### DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes

**Authors**: Ming-Hsuan Yang, Deqing Sun, Yongtao Wang, Xiaojun Shan, Zhiwei Lin, Xiaoyu Zhou

<details>
<summary><b>Abstract</b></summary>
We present DrivingGaussian, an efficient and effective framework for surrounding dynamic autonomous driving scenes. For complex scenes with moving objects, we first sequentially and progressively model the static background of the entire scene with incremental static 3D Gaussians. We then leverage a composite dynamic Gaussian graph to handle multiple moving objects, individually reconstructing each object and restoring their accurate positions and occlusion relationships within the scene. We further use a LiDAR prior for Gaussian Splatting to reconstruct scenes with greater details and maintain panoramic consistency. DrivingGaussian outperforms existing methods in dynamic driving scene reconstruction and enables photorealistic surround-view synthesis with high-fidelity and multi-camera consistency. Our project page is at: https://github.com/VDIGPKU/DrivingGaussian.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.07920v3.pdf)

## Adaptive Control
### Densification
#### HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

**Authors**: Ziwei Liu, Xihui Liu, Dahua Lin, Gang Zeng, Ying Shan, Jiaxiang Tang, Xiaohang Zhan, Xian Liu

<details>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17061v2.pdf)

#### Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting

**Authors**: Hengshuang Zhao, Tong He, Yixing Lao, WenBo Hu, Zheng Zhang

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results while advancing real-time rendering performance. However, it relies heavily on the quality of the initial point cloud, resulting in blurring and needle-like artifacts in areas with insufficient initializing points. This is mainly attributed to the point cloud growth condition in 3DGS that only considers the average gradient magnitude of points from observable views, thereby failing to grow for large Gaussians that are observable for many viewpoints while many of them are only covered in the boundaries. To this end, we propose a novel method, named Pixel-GS, to take into account the number of pixels covered by the Gaussian in each view during the computation of the growth condition. We regard the covered pixel numbers as the weights to dynamically average the gradients from different views, such that the growth of large Gaussians can be prompted. As a result, points within the areas with insufficient initializing points can be grown more effectively, leading to a more accurate and detailed reconstruction. In addition, we propose a simple yet effective strategy to scale the gradient field according to the distance to the camera, to suppress the growth of floaters near the camera. Extensive experiments both qualitatively and quantitatively demonstrate that our method achieves state-of-the-art rendering quality while maintaining real-time rendering speed, on the challenging Mip-NeRF 360 and Tanks & Temples datasets.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.15530v1.pdf)

#### A New Split Algorithm for 3D Gaussian Splatting

**Authors**: 

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting models, as a novel explicit 3D representation, have been applied in many domains recently, such as explicit geometric editing and geometry generation. Progress has been rapid. However, due to their mixed scales and cluttered shapes, 3D Gaussian splatting models can produce a blurred or needle-like effect near the surface. At the same time, 3D Gaussian splatting models tend to flatten large untextured regions, yielding a very sparse point cloud. These problems are caused by the non-uniform nature of 3D Gaussian splatting models, so in this paper, we propose a new 3D Gaussian splitting algorithm, which can produce a more uniform and surface-bounded 3D Gaussian splatting model. Our algorithm splits an $N$-dimensional Gaussian into two N-dimensional Gaussians. It ensures consistency of mathematical characteristics and similarity of appearance, allowing resulting 3D Gaussian splatting models to be more uniform and a better fit to the underlying surface, and thus more suitable for explicit editing, point cloud extraction and other tasks. Meanwhile, our 3D Gaussian splitting approach has a very simple closed-form solution, making it readily applicable to any 3D Gaussian model.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.09143v1.pdf)

#### 3DGStream: On-the-Fly Training of 3D Gaussians for Efficient Streaming of Photo-Realistic Free-Viewpoint Videos

**Authors**: Wei Xing, Lei Zhao, Zhanjie Zhang, Guangyuan Li, Han Jiao, Jiakai Sun

<details>
<summary><b>Abstract</b></summary>
Constructing photo-realistic Free-Viewpoint Videos (FVVs) of dynamic scenes from multi-view videos remains a challenging endeavor. Despite the remarkable advancements achieved by current neural rendering techniques, these methods generally require complete video sequences for offline training and are not capable of real-time rendering. To address these constraints, we introduce 3DGStream, a method designed for efficient FVV streaming of real-world dynamic scenes. Our method achieves fast on-the-fly per-frame reconstruction within 12 seconds and real-time rendering at 200 FPS. Specifically, we utilize 3D Gaussians (3DGs) to represent the scene. Instead of the na\"ive approach of directly optimizing 3DGs per-frame, we employ a compact Neural Transformation Cache (NTC) to model the translations and rotations of 3DGs, markedly reducing the training time and storage required for each FVV frame. Furthermore, we propose an adaptive 3DG addition strategy to handle emerging objects in dynamic scenes. Experiments demonstrate that 3DGStream achieves competitive performance in terms of rendering speed, image quality, training time, and model storage when compared with state-of-the-art methods.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.01444v4.pdf)

#### DeblurGS: Gaussian Splatting for Camera Motion Blur

**Authors**: Kyoung Mu Lee, Dongwoo Lee, JaeYoung Chung, Jeongtaek Oh

<details>
<summary><b>Abstract</b></summary>
Although significant progress has been made in reconstructing sharp 3D scenes from motion-blurred images, a transition to real-world applications remains challenging. The primary obstacle stems from the severe blur which leads to inaccuracies in the acquisition of initial camera poses through Structure-from-Motion, a critical aspect often overlooked by previous approaches. To address this challenge, we propose DeblurGS, a method to optimize sharp 3D Gaussian Splatting from motion-blurred images, even with the noisy camera pose initialization. We restore a fine-grained sharp scene by leveraging the remarkable reconstruction capability of 3D Gaussian Splatting. Our approach estimates the 6-Degree-of-Freedom camera motion for each blurry observation and synthesizes corresponding blurry renderings for the optimization process. Furthermore, we propose Gaussian Densification Annealing strategy to prevent the generation of inaccurate Gaussians at erroneous locations during the early training stages when camera motion is still imprecise. Comprehensive experiments demonstrate that our DeblurGS achieves state-of-the-art performance in deblurring and novel view synthesis for real-world and synthetic benchmark datasets, as well as field-captured blurry smartphone videos.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.11358v2.pdf)

#### FDGaussian: Fast Gaussian Splatting from Single Image via Geometric-aware Diffusion Model

**Authors**: Yu-Gang Jiang, Zuxuan Wu, Zhen Xing, Qijun Feng

<details>
<summary><b>Abstract</b></summary>
Reconstructing detailed 3D objects from single-view images remains a challenging task due to the limited information available. In this paper, we introduce FDGaussian, a novel two-stage framework for single-image 3D reconstruction. Recent methods typically utilize pre-trained 2D diffusion models to generate plausible novel views from the input image, yet they encounter issues with either multi-view inconsistency or lack of geometric fidelity. To overcome these challenges, we propose an orthogonal plane decomposition mechanism to extract 3D geometric features from the 2D input, enabling the generation of consistent multi-view images. Moreover, we further accelerate the state-of-the-art Gaussian Splatting incorporating epipolar attention to fuse images from different viewpoints. We demonstrate that FDGaussian generates images with high consistency across different views and reconstructs high-quality 3D objects, both qualitatively and quantitatively. More examples can be found at our website https://qjfeng.net/FDGaussian/.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.10242v1.pdf)

#### FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

**Authors**: Zhangyang Wang, Yifan Jiang, Zhiwen Fan, Zehao Zhu

<details>
<summary><b>Abstract</b></summary>
Novel view synthesis from limited observations remains an important and persistent task. However, high efficiency in existing NeRF-based few-shot view synthesis is often compromised to obtain an accurate 3D representation. To address this challenge, we propose a few-shot view synthesis framework based on 3D Gaussian Splatting that enables real-time and photo-realistic view synthesis with as few as three training views. The proposed method, dubbed FSGS, handles the extremely sparse initialized SfM points with a thoughtfully designed Gaussian Unpooling process. Our method iteratively distributes new Gaussians around the most representative locations, subsequently infilling local details in vacant areas. We also integrate a large-scale pre-trained monocular depth estimator within the Gaussians optimization process, leveraging online augmented views to guide the geometric optimization towards an optimal solution. Starting from sparse points observed from limited input viewpoints, our FSGS can accurately grow into unseen regions, comprehensively covering the scene and boosting the rendering quality of novel views. Overall, FSGS achieves state-of-the-art performance in both accuracy and rendering efficiency across diverse datasets, including LLFF, Mip-NeRF360, and Blender. Project website: https://zehaozhu.github.io/FSGS/.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.00451v2.pdf)

#### GVGEN: Text-to-3D Generation with Volumetric Representation

**Authors**: Tong He, Wanli Ouyang, Chun Yuan, Xiaoshui Huang, Yangguang Li, Di Huang, Sida Peng, Junyi Chen, Xianglong He

<details>
<summary><b>Abstract</b></summary>
In recent years, 3D Gaussian splatting has emerged as a powerful technique for 3D reconstruction and generation, known for its fast and high-quality rendering capabilities. To address these shortcomings, this paper introduces a novel diffusion-based framework, GVGEN, designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques:(1) Structured Volumetric Representation. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. To better optimize the representation of these details, we propose a unique pruning and densifying method named the Candidate Pool Strategy, enhancing detail fidelity through selective optimization. (2) Coarse-to-fine Generation Pipeline. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. Our framework, GVGEN, demonstrates superior performance in qualitative and quantitative assessments compared to existing 3D generation methods. Simultaneously, it maintains a fast generation speed ($\sim$7 seconds), effectively striking a balance between quality and efficiency. Our project page is: https://gvgen.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12957v2.pdf)

### Pruning
#### LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS

**Authors**: Zhangyang Wang, Dejia Xu, Zehao Zhu, Kairun Wen, Kevin Wang, Zhiwen Fan

<details>
<summary><b>Abstract</b></summary>
Recent advancements in real-time neural rendering using point-based techniques have paved the way for the widespread adoption of 3D representations. However, foundational approaches like 3D Gaussian Splatting come with a substantial storage overhead caused by growing the SfM points to millions, often demanding gigabyte-level disk space for a single unbounded scene, posing significant scalability challenges and hindering the splatting efficiency. To address this challenge, we introduce LightGaussian, a novel method designed to transform 3D Gaussians into a more efficient and compact format. Drawing inspiration from the concept of Network Pruning, LightGaussian identifies Gaussians that are insignificant in contributing to the scene reconstruction and adopts a pruning and recovery process, effectively reducing redundancy in Gaussian counts while preserving visual effects. Additionally, LightGaussian employs distillation and pseudo-view augmentation to distill spherical harmonics to a lower degree, allowing knowledge transfer to more compact representations while maintaining reflectance. Furthermore, we propose a hybrid scheme, VecTree Quantization, to quantize all attributes, resulting in lower bitwidth representations with minimal accuracy losses. In summary, LightGaussian achieves an averaged compression rate over 15x while boosting the FPS from 139 to 215, enabling an efficient representation of complex scenes on Mip-NeRF 360, Tank and Temple datasets. Project website: https://lightgaussian.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17245v5.pdf)

#### GauHuman: Articulated Gaussian Splatting from Monocular Human Videos

**Authors**: Ziwei Liu, Shoukang Hu

<details>
<summary><b>Abstract</b></summary>
We present, GauHuman, a 3D human model with Gaussian Splatting for both fast training (1 ~ 2 minutes) and real-time rendering (up to 189 FPS), compared with existing NeRF-based implicit representation modelling frameworks demanding hours of training and seconds of rendering per frame. Specifically, GauHuman encodes Gaussian Splatting in the canonical space and transforms 3D Gaussians from canonical space to posed space with linear blend skinning (LBS), in which effective pose and LBS refinement modules are designed to learn fine details of 3D humans under negligible computational cost. Moreover, to enable fast optimization of GauHuman, we initialize and prune 3D Gaussians with 3D human prior, while splitting/cloning via KL divergence guidance, along with a novel merge operation for further speeding up. Extensive experiments on ZJU_Mocap and MonoCap datasets demonstrate that GauHuman achieves state-of-the-art performance quantitatively and qualitatively with fast training and real-time rendering speed. Notably, without sacrificing rendering quality, GauHuman can fast model the 3D human performer with ~13k 3D Gaussians.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02973v1.pdf)

#### Gaussian Splatting SLAM

**Authors**: Andrew J. Davison, Paul H. J. Kelly, Riku Murai, Hidenobu Matsuki

<details>
<summary><b>Abstract</b></summary>
We present the first application of 3D Gaussian Splatting in monocular SLAM, the most fundamental but the hardest setup for Visual SLAM. Our method, which runs live at 3fps, utilises Gaussians as the only 3D representation, unifying the required representation for accurate, efficient tracking, mapping, and high-quality rendering. Designed for challenging monocular settings, our approach is seamlessly extendable to RGB-D SLAM when an external depth sensor is available. Several innovations are required to continuously reconstruct 3D scenes with high fidelity from a live camera. First, to move beyond the original 3DGS algorithm, which requires accurate poses from an offline Structure from Motion (SfM) system, we formulate camera tracking for 3DGS using direct optimisation against the 3D Gaussians, and show that this enables fast and robust tracking with a wide basin of convergence. Second, by utilising the explicit nature of the Gaussians, we introduce geometric verification and regularisation to handle the ambiguities occurring in incremental 3D dense reconstruction. Finally, we introduce a full SLAM system which not only achieves state-of-the-art results in novel view synthesis and trajectory estimation but also reconstruction of tiny and even transparent objects.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.06741v2.pdf)

#### GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction

**Authors**: Bo Dai, Yuanbo Xiangli, Lihan Jiang, Linning Xu, Tao Lu, Mulin Yu

<details>
<summary><b>Abstract</b></summary>
Presenting a 3D scene from multiview images remains a core and long-standing challenge in computer vision and computer graphics. Two main requirements lie in rendering and reconstruction. Notably, SOTA rendering quality is usually achieved with neural volumetric rendering techniques, which rely on aggregated point/primitive-wise color and neglect the underlying scene geometry. Learning of neural implicit surfaces is sparked from the success of neural rendering. Current works either constrain the distribution of density fields or the shape of primitives, resulting in degraded rendering quality and flaws on the learned scene surfaces. The efficacy of such methods is limited by the inherent constraints of the chosen neural representation, which struggles to capture fine surface details, especially for larger, more intricate scenes. To address these issues, we introduce GSDF, a novel dual-branch architecture that combines the benefits of a flexible and efficient 3D Gaussian Splatting (3DGS) representation with neural Signed Distance Fields (SDF). The core idea is to leverage and enhance the strengths of each branch while alleviating their limitation through mutual guidance and joint supervision. We show on diverse scenes that our design unlocks the potential for more accurate and detailed surface reconstructions, and at the meantime benefits 3DGS rendering with structures that are more aligned with the underlying geometry.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.16964v1.pdf)

#### GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting

**Authors**: Dong Wang, Bin Zhao, Xuelong Li, Zhigang Wang, Dan Xu, Delin Qu, Chi Yan

<details>
<summary><b>Abstract</b></summary>
In this paper, we introduce \textbf{GS-SLAM} that first utilizes 3D Gaussian representation in the Simultaneous Localization and Mapping (SLAM) system. It facilitates a better balance between efficiency and accuracy. Compared to recent SLAM methods employing neural implicit representations, our method utilizes a real-time differentiable splatting rendering pipeline that offers significant speedup to map optimization and RGB-D rendering. Specifically, we propose an adaptive expansion strategy that adds new or deletes noisy 3D Gaussians in order to efficiently reconstruct new observed scene geometry and improve the mapping of previously observed areas. This strategy is essential to extend 3D Gaussian representation to reconstruct the whole scene rather than synthesize a static object in existing methods. Moreover, in the pose tracking process, an effective coarse-to-fine technique is designed to select reliable 3D Gaussian representations to optimize camera pose, resulting in runtime reduction and robust estimation. Our method achieves competitive performance compared with existing state-of-the-art real-time methods on the Replica, TUM-RGBD datasets. Project page: https://gs-slam.github.io/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.11700v4.pdf)

#### NEDS-SLAM: A Novel Neural Explicit Dense Semantic SLAM Framework using 3D Gaussian Splatting

**Authors**: Zongwu Xie, Boyu Ma, Guanghu Xie, Yang Liu, Yiming Ji

<details>
<summary><b>Abstract</b></summary>
We propose NEDS-SLAM, an Explicit Dense semantic SLAM system based on 3D Gaussian representation, that enables robust 3D semantic mapping, accurate camera tracking, and high-quality rendering in real-time. In the system, we propose a Spatially Consistent Feature Fusion model to reduce the effect of erroneous estimates from pre-trained segmentation head on semantic reconstruction, achieving robust 3D semantic Gaussian mapping. Additionally, we employ a lightweight encoder-decoder to compress the high-dimensional semantic features into a compact 3D Gaussian representation, mitigating the burden of excessive memory consumption. Furthermore, we leverage the advantage of 3D Gaussian splatting, which enables efficient and differentiable novel view rendering, and propose a Virtual Camera View Pruning method to eliminate outlier GS points, thereby effectively enhancing the quality of scene representations. Our NEDS-SLAM method demonstrates competitive performance over existing dense semantic SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in 3D dense semantic mapping.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11679v2.pdf)

#### Compact 3D Gaussian Representation for Radiance Field

**Authors**: Eunbyung Park, Jong Hwan Ko, Xiangyu Sun, Daniel Rho, Joo Chan Lee

<details>
<summary><b>Abstract</b></summary>
Neural Radiance Fields (NeRFs) have demonstrated remarkable potential in capturing complex 3D scenes with high fidelity. However, one persistent challenge that hinders the widespread adoption of NeRFs is the computational bottleneck due to the volumetric rendering. On the other hand, 3D Gaussian splatting (3DGS) has recently emerged as an alternative representation that leverages a 3D Gaussisan-based representation and adopts the rasterization pipeline to render the images rather than volumetric rendering, achieving very fast rendering speed and promising image quality. However, a significant drawback arises as 3DGS entails a substantial number of 3D Gaussians to maintain the high fidelity of the rendered images, which requires a large amount of memory and storage. To address this critical issue, we place a specific emphasis on two key objectives: reducing the number of Gaussian points without sacrificing performance and compressing the Gaussian attributes, such as view-dependent color and covariance. To this end, we propose a learnable mask strategy that significantly reduces the number of Gaussians while preserving high performance. In addition, we propose a compact but effective representation of view-dependent color by employing a grid-based neural field rather than relying on spherical harmonics. Finally, we learn codebooks to compactly represent the geometric attributes of Gaussian by vector quantization. With model compression techniques such as quantization and entropy coding, we consistently show over 25$\times$ reduced storage and enhanced rendering speed, while maintaining the quality of the scene representation, compared to 3DGS. Our work provides a comprehensive framework for 3D scene representation, achieving high performance, fast training, compactness, and real-time rendering. Our project page is available at https://maincold2.github.io/c3dgs/.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.13681v2.pdf)

## Post-Processing
#### LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

**Authors**: Ziwei Liu, Gang Zeng, Tengfei Wang, Xiaokang Chen, Zhaoxi Chen, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
3D content creation has achieved significant progress in terms of both quality and speed. Although current feed-forward models can produce 3D objects in seconds, their resolution is constrained by the intensive computation required during training. In this paper, we introduce Large Multi-View Gaussian Model (LGM), a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models. Extensive experiments demonstrate the high fidelity and efficiency of our approach. Notably, we maintain the fast speed to generate 3D objects within 5 seconds while boosting the training resolution to 512, thereby achieving high-resolution 3D content generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.05054v1.pdf)

#### GGRt: Towards Generalizable 3D Gaussians without Pose Priors in Real-Time
  - **Not Found on Arxiv by arxiv api, will be append manually later**

#### SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

**Authors**: Vincent Lepetit, Antoine Guédon

<details>
<summary><b>Abstract</b></summary>
We propose a method to allow precise and extremely fast mesh extraction from 3D Gaussian Splatting. Gaussian Splatting has recently become very popular as it yields realistic rendering while being significantly faster to train than NeRFs. It is however challenging to extract a mesh from the millions of tiny 3D gaussians as these gaussians tend to be unorganized after optimization and no method has been proposed so far. Our first key contribution is a regularization term that encourages the gaussians to align well with the surface of the scene. We then introduce a method that exploits this alignment to extract a mesh from the Gaussians using Poisson reconstruction, which is fast, scalable, and preserves details, in contrast to the Marching Cubes algorithm usually applied to extract meshes from Neural SDFs. Finally, we introduce an optional refinement strategy that binds gaussians to the surface of the mesh, and jointly optimizes these Gaussians and the mesh through Gaussian splatting rendering. This enables easy editing, sculpting, rigging, animating, compositing and relighting of the Gaussians using traditional softwares by manipulating the mesh instead of the gaussians themselves. Retrieving such an editable mesh for realistic rendering is done within minutes with our method, compared to hours with the state-of-the-art methods on neural SDFs, while providing a better rendering quality. Our project page is the following: https://anttwo.github.io/sugar/
</details>

[📄 Paper](https://arxiv.org/pdf/2311.12775v3.pdf)

#### Identifying Unnecessary 3D Gaussians using Clustering for Fast Rendering of 3D Gaussian Splatting

**Authors**: Jongsun Park, Hyeongwon Kim, Joongho Jo

<details>
<summary><b>Abstract</b></summary>
3D Gaussian splatting (3D-GS) is a new rendering approach that outperforms the neural radiance field (NeRF) in terms of both speed and image quality. 3D-GS represents 3D scenes by utilizing millions of 3D Gaussians and projects these Gaussians onto the 2D image plane for rendering. However, during the rendering process, a substantial number of unnecessary 3D Gaussians exist for the current view direction, resulting in significant computation costs associated with their identification. In this paper, we propose a computational reduction technique that quickly identifies unnecessary 3D Gaussians in real-time for rendering the current view without compromising image quality. This is accomplished through the offline clustering of 3D Gaussians that are close in distance, followed by the projection of these clusters onto a 2D image plane during runtime. Additionally, we analyze the bottleneck associated with the proposed technique when executed on GPUs and propose an efficient hardware architecture that seamlessly supports the proposed scheme. For the Mip-NeRF360 dataset, the proposed technique excludes 63% of 3D Gaussians on average before the 2D image projection, which reduces the overall rendering computation by almost 38.3% without sacrificing peak-signal-to-noise-ratio (PSNR). The proposed accelerator also achieves a speedup of 10.7x compared to a GPU.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.13827v1.pdf)

#### Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement

**Authors**: Gang Zeng, Jingdong Wang, Errui Ding, Tianshu Hu, Xiaokang Chen, Hang Zhou, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
Neural Radiance Fields (NeRF) have constituted a remarkable breakthrough in image-based 3D reconstruction. However, their implicit volumetric representations differ significantly from the widely-adopted polygonal meshes and lack support from common 3D software and hardware, making their rendering and manipulation inefficient. To overcome this limitation, we present a novel framework that generates textured surface meshes from images. Our approach begins by efficiently initializing the geometry and view-dependency decomposed appearance with a NeRF. Subsequently, a coarse mesh is extracted, and an iterative surface refining algorithm is developed to adaptively adjust both vertex positions and face density based on re-projected rendering errors. We jointly refine the appearance with geometry and bake it into texture images for real-time rendering. Extensive experiments demonstrate that our method achieves superior mesh quality and competitive rendering quality.
</details>

[📄 Paper](https://arxiv.org/pdf/2303.02091v2.pdf)

#### SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing

**Authors**: Hao Zhao, Weihao Gu, Xiang He, Jingwei Zhao, Huan-ang Gao, Shiran Yuan, Jv Zheng, Xiaowei Song

<details>
<summary><b>Abstract</b></summary>
In this paper, we present a Scale-adaptive method for Anti-aliasing Gaussian Splatting (SA-GS). While the state-of-the-art method Mip-Splatting needs modifying the training procedure of Gaussian splatting, our method functions at test-time and is training-free. Specifically, SA-GS can be applied to any pretrained Gaussian splatting field as a plugin to significantly improve the field's anti-alising performance. The core technique is to apply 2D scale-adaptive filters to each Gaussian during test time. As pointed out by Mip-Splatting, observing Gaussians at different frequencies leads to mismatches between the Gaussian scales during training and testing. Mip-Splatting resolves this issue using 3D smoothing and 2D Mip filters, which are unfortunately not aware of testing frequency. In this work, we show that a 2D scale-adaptive filter that is informed of testing frequency can effectively match the Gaussian scale, thus making the Gaussian primitive distribution remain consistent across different testing frequencies. When scale inconsistency is eliminated, sampling rates smaller than the scene frequency result in conventional jaggedness, and we propose to integrate the projected 2D Gaussian within each pixel during testing. This integration is actually a limiting case of super-sampling, which significantly improves anti-aliasing performance over vanilla Gaussian Splatting. Through extensive experiments using various settings and both bounded and unbounded scenes, we show SA-GS performs comparably with or better than Mip-Splatting. Note that super-sampling and integration are only effective when our scale-adaptive filtering is activated. Our codes, data and models are available at https://github.com/zsy1987/SA-GS.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.19615v1.pdf)

#### Augmented Reality for Depth Cues in Monocular Minimally Invasive Surgery

**Authors**: Jian Jun Zhang, Long Chen, Wen Tang, Nigel W. John, Tao Ruan Wan

<details>
<summary><b>Abstract</b></summary>
One of the major challenges in Minimally Invasive Surgery (MIS) such as
laparoscopy is the lack of depth perception. In recent years, laparoscopic
scene tracking and surface reconstruction has been a focus of investigation to
provide rich additional information to aid the surgical process and compensate
for the depth perception issue. However, robust 3D surface reconstruction and
augmented reality with depth perception on the reconstructed scene are yet to
be reported. This paper presents our work in this area. First, we adopt a
state-of-the-art visual simultaneous localization and mapping (SLAM) framework
- ORB-SLAM - and extend the algorithm for use in MIS scenes for reliable
endoscopic camera tracking and salient point mapping. We then develop a robust
global 3D surface reconstruction frame- work based on the sparse point clouds
extracted from the SLAM framework. Our approach is to combine an outlier
removal filter within a Moving Least Squares smoothing algorithm and then
employ Poisson surface reconstruction to obtain smooth surfaces from the
unstructured sparse point cloud. Our proposed method has been quantitatively
evaluated compared with ground-truth camera trajectories and the organ model
surface we used to render the synthetic simulation videos. In vivo laparoscopic
videos used in the tests have demonstrated the robustness and accuracy of our
proposed framework on both camera tracking and surface reconstruction,
illustrating the potential of our algorithm for depth augmentation and
depth-corrected augmented reality in MIS with monocular endoscopes.
</details>

[📄 Paper](http://arxiv.org/pdf/1703.01243v1.pdf)

#### Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes

**Authors**: Andreas Geiger, Torsten Sattler, Zehao Yu

<details>
<summary><b>Abstract</b></summary>
Recently, 3D Gaussian Splatting (3DGS) has demonstrated impressive novel view synthesis results, while allowing the rendering of high-resolution images in real-time. However, leveraging 3D Gaussians for surface reconstruction poses significant challenges due to the explicit and disconnected nature of 3D Gaussians. In this work, we present Gaussian Opacity Fields (GOF), a novel approach for efficient, high-quality, and compact surface reconstruction in unbounded scenes. Our GOF is derived from ray-tracing-based volume rendering of 3D Gaussians, enabling direct geometry extraction from 3D Gaussians by identifying its levelset, without resorting to Poisson reconstruction or TSDF fusion as in previous work. We approximate the surface normal of Gaussians as the normal of the ray-Gaussian intersection plane, enabling the application of regularization that significantly enhances geometry. Furthermore, we develop an efficient geometry extraction method utilizing marching tetrahedra, where the tetrahedral grids are induced from 3D Gaussians and thus adapt to the scene's complexity. Our evaluations reveal that GOF surpasses existing 3DGS-based methods in surface reconstruction and novel view synthesis. Further, it compares favorably to, or even outperforms, neural implicit methods in both quality and speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.10772v1.pdf)

## Integration with Other Representations
### Point Clouds
#### GaussNav: Gaussian Splatting for Visual Navigation

**Authors**: Houqiang Li, Wengang Zhou, Min Wang, Xiaohan Lei

<details>
<summary><b>Abstract</b></summary>
In embodied vision, Instance ImageGoal Navigation (IIN) requires an agent to locate a specific object depicted in a goal image within an unexplored environment. The primary difficulty of IIN stems from the necessity of recognizing the target object across varying viewpoints and rejecting potential distractors. Existing map-based navigation methods largely adopt the representation form of Bird's Eye View (BEV) maps, which, however, lack the representation of detailed textures in a scene. To address the above issues, we propose a new Gaussian Splatting Navigation (abbreviated as GaussNav) framework for IIN task, which constructs a novel map representation based on 3D Gaussian Splatting (3DGS). The proposed framework enables the agent to not only memorize the geometry and semantic information of the scene, but also retain the textural features of objects. Our GaussNav framework demonstrates a significant leap in performance, evidenced by an increase in Success weighted by Path Length (SPL) from 0.252 to 0.578 on the challenging Habitat-Matterport 3D (HM3D) dataset. Our code will be made publicly available.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11625v2.pdf)

### Mesh
#### Bridging 3D Gaussian and Mesh for Freeview Video Rendering

**Authors**: Shenghua Gao, Yujun Shen, Minghui Yang, Nan Xue, Yanbo Fan, Hongrui Cai, Jiafei Li, Xuan Wang, Yuting Xiao

<details>
<summary><b>Abstract</b></summary>
This is only a preview version of GauMesh. Recently, primitive-based rendering has been proven to achieve convincing results in solving the problem of modeling and rendering the 3D dynamic scene from 2D images. Despite this, in the context of novel view synthesis, each type of primitive has its inherent defects in terms of representation ability. It is difficult to exploit the mesh to depict the fuzzy geometry. Meanwhile, the point-based splatting (e.g. the 3D Gaussian Splatting) method usually produces artifacts or blurry pixels in the area with smooth geometry and sharp textures. As a result, it is difficult, even not impossible, to represent the complex and dynamic scene with a single type of primitive. To this end, we propose a novel approach, GauMesh, to bridge the 3D Gaussian and Mesh for modeling and rendering the dynamic scenes. Given a sequence of tracked mesh as initialization, our goal is to simultaneously optimize the mesh geometry, color texture, opacity maps, a set of 3D Gaussians, and the deformation field. At a specific time, we perform $\alpha$-blending on the RGB and opacity values based on the merged and re-ordered z-buffers from mesh and 3D Gaussian rasterizations. This produces the final rendering, which is supervised by the ground-truth image. Experiments demonstrate that our approach adapts the appropriate type of primitives to represent the different parts of the dynamic scene and outperforms all the baseline methods in both quantitative and qualitative comparisons without losing render speed.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.11453v1.pdf)

#### LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation

**Authors**: Ziwei Liu, Gang Zeng, Tengfei Wang, Xiaokang Chen, Zhaoxi Chen, Jiaxiang Tang

<details>
<summary><b>Abstract</b></summary>
3D content creation has achieved significant progress in terms of both quality and speed. Although current feed-forward models can produce 3D objects in seconds, their resolution is constrained by the intensive computation required during training. In this paper, we introduce Large Multi-View Gaussian Model (LGM), a novel framework designed to generate high-resolution 3D models from text prompts or single-view images. Our key insights are two-fold: 1) 3D Representation: We propose multi-view Gaussian features as an efficient yet powerful representation, which can then be fused together for differentiable rendering. 2) 3D Backbone: We present an asymmetric U-Net as a high-throughput backbone operating on multi-view images, which can be produced from text or single-view image input by leveraging multi-view diffusion models. Extensive experiments demonstrate the high fidelity and efficiency of our approach. Notably, we maintain the fast speed to generate 3D objects within 5 seconds while boosting the training resolution to 512, thereby achieving high-resolution 3D content generation.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.05054v1.pdf)

### Triplane
#### Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

**Authors**: Song-Hai Zhang, Yan-Pei Cao, Ding Liang, Yangguang Li, Yuan-Chen Guo, Zhipeng Yu, Zi-Xin Zou

<details>
<summary><b>Abstract</b></summary>
Recent advancements in 3D reconstruction from single images have been driven by the evolution of generative models. Prominent among these are methods based on Score Distillation Sampling (SDS) and the adaptation of diffusion models in the 3D domain. Despite their progress, these techniques often face limitations due to slow optimization or rendering processes, leading to extensive training and optimization times. In this paper, we introduce a novel approach for single-view reconstruction that efficiently generates a 3D model from a single image via feed-forward inference. Our method utilizes two transformer-based networks, namely a point decoder and a triplane decoder, to reconstruct 3D objects using a hybrid Triplane-Gaussian intermediate representation. This hybrid representation strikes a balance, achieving a faster rendering speed compared to implicit representations while simultaneously delivering superior rendering quality than explicit representations. The point decoder is designed for generating point clouds from single images, offering an explicit representation which is then utilized by the triplane decoder to query Gaussian features for each point. This design choice addresses the challenges associated with directly regressing explicit 3D Gaussian attributes characterized by their non-structural nature. Subsequently, the 3D Gaussians are decoded by an MLP to enable rapid rendering through splatting. Both decoders are built upon a scalable, transformer-based architecture and have been efficiently trained on large-scale 3D datasets. The evaluations conducted on both synthetic datasets and real-world images demonstrate that our method not only achieves higher quality but also ensures a faster runtime in comparison to previous state-of-the-art techniques. Please see our project page at https://zouzx.github.io/TriplaneGaussian/.
</details>
[📄 Paper](https://arxiv.org/pdf/2312.09147v2.pdf)

#### Control4d: Dynamic portrait editing by learning 4d gan from 2d diffusion-based editor
  - **Not Found on Arxiv by arxiv api, will be append manually later**

### Grid
#### Compact 3D Scene Representation via Self-Organizing Gaussian Grids

**Authors**: Peter Eisert, Anna Hilsmann, Florian Barthel, Wieland Morgenstern

<details>
<summary><b>Abstract</b></summary>
3D Gaussian Splatting has recently emerged as a highly promising technique for modeling of static 3D scenes. In contrast to Neural Radiance Fields, it utilizes efficient rasterization allowing for very fast rendering at high-quality. However, the storage size is significantly higher, which hinders practical deployment, e.g. on resource constrained devices. In this paper, we introduce a compact scene representation organizing the parameters of 3D Gaussian Splatting (3DGS) into a 2D grid with local homogeneity, ensuring a drastic reduction in storage requirements without compromising visual quality during rendering. Central to our idea is the explicit exploitation of perceptual redundancies present in natural scenes. In essence, the inherent nature of a scene allows for numerous permutations of Gaussian parameters to equivalently represent it. To this end, we propose a novel highly parallel algorithm that regularly arranges the high-dimensional Gaussian parameters into a 2D grid while preserving their neighborhood structure. During training, we further enforce local smoothness between the sorted parameters in the grid. The uncompressed Gaussians use the same structure as 3DGS, ensuring a seamless integration with established renderers. Our method achieves a reduction factor of 17x to 42x in size for complex scenes with no increase in training time, marking a substantial leap forward in the domain of 3D scene distribution and consumption. Additional information can be found on our project page: https://fraunhoferhhi.github.io/Self-Organizing-Gaussians/
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13299v2.pdf)

### Implicit Representation
#### 3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting

**Authors**: Xiaojuan Qi, Jiangmiao Pang, Yilun Chen, ZiYi Yang, Xiuzhe Wu, Yi-Hua Huang, Yang-tian Sun, Xiaoyang Lyu

<details>
<summary><b>Abstract</b></summary>
In this paper, we present an implicit surface reconstruction method with 3D Gaussian Splatting (3DGS), namely 3DGSR, that allows for accurate 3D reconstruction with intricate details while inheriting the high efficiency and rendering quality of 3DGS. The key insight is incorporating an implicit signed distance field (SDF) within 3D Gaussians to enable them to be aligned and jointly optimized. First, we introduce a differentiable SDF-to-opacity transformation function that converts SDF values into corresponding Gaussians' opacities. This function connects the SDF and 3D Gaussians, allowing for unified optimization and enforcing surface constraints on the 3D Gaussians. During learning, optimizing the 3D Gaussians provides supervisory signals for SDF learning, enabling the reconstruction of intricate details. However, this only provides sparse supervisory signals to the SDF at locations occupied by Gaussians, which is insufficient for learning a continuous SDF. Then, to address this limitation, we incorporate volumetric rendering and align the rendered geometric attributes (depth, normal) with those derived from 3D Gaussians. This consistency regularization introduces supervisory signals to locations not covered by discrete 3D Gaussians, effectively eliminating redundant surfaces outside the Gaussian sampling range. Our extensive experimental results demonstrate that our 3DGSR method enables high-quality 3D surface reconstruction while preserving the efficiency and rendering quality of 3DGS. Besides, our method competes favorably with leading surface reconstruction techniques while offering a more efficient learning process and much better rendering qualities. The code will be available at https://github.com/CVMI-Lab/3DGSR.
</details>

[📄 Paper](https://arxiv.org/pdf/2404.00409v1.pdf)

#### GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction

**Authors**: Bo Dai, Yuanbo Xiangli, Lihan Jiang, Linning Xu, Tao Lu, Mulin Yu

<details>
<summary><b>Abstract</b></summary>
Presenting a 3D scene from multiview images remains a core and long-standing challenge in computer vision and computer graphics. Two main requirements lie in rendering and reconstruction. Notably, SOTA rendering quality is usually achieved with neural volumetric rendering techniques, which rely on aggregated point/primitive-wise color and neglect the underlying scene geometry. Learning of neural implicit surfaces is sparked from the success of neural rendering. Current works either constrain the distribution of density fields or the shape of primitives, resulting in degraded rendering quality and flaws on the learned scene surfaces. The efficacy of such methods is limited by the inherent constraints of the chosen neural representation, which struggles to capture fine surface details, especially for larger, more intricate scenes. To address these issues, we introduce GSDF, a novel dual-branch architecture that combines the benefits of a flexible and efficient 3D Gaussian Splatting (3DGS) representation with neural Signed Distance Fields (SDF). The core idea is to leverage and enhance the strengths of each branch while alleviating their limitation through mutual guidance and joint supervision. We show on diverse scenes that our design unlocks the potential for more accurate and detailed surface reconstructions, and at the meantime benefits 3DGS rendering with structures that are more aligned with the underlying geometry.
</details>

[📄 Paper](https://arxiv.org/pdf/2403.16964v1.pdf)

#### Gaussian Splatting with NeRF-based Color and Opacity

**Authors**: Przemysław Spurek, Sławomir Tadeja, Jacek Tabor, Weronika Smolak, Dawid Malarz

<details>
<summary><b>Abstract</b></summary>
Neural Radiance Fields (NeRFs) have demonstrated the remarkable potential of neural networks to capture the intricacies of 3D objects. By encoding the shape and color information within neural network weights, NeRFs excel at producing strikingly sharp novel views of 3D objects. Recently, numerous generalizations of NeRFs utilizing generative models have emerged, expanding its versatility. In contrast, Gaussian Splatting (GS) offers a similar render quality with faster training and inference as it does not need neural networks to work. It encodes information about the 3D objects in the set of Gaussian distributions that can be rendered in 3D similarly to classical meshes. Unfortunately, GS are difficult to condition since they usually require circa hundred thousand Gaussian components. To mitigate the caveats of both models, we propose a hybrid model Viewing Direction Gaussian Splatting (VDGS) that uses GS representation of the 3D object's shape and NeRF-based encoding of color and opacity. Our model uses Gaussian distributions with trainable positions (i.e. means of Gaussian), shape (i.e. covariance of Gaussian), color and opacity, and a neural network that takes Gaussian parameters and viewing direction to produce changes in the said color and opacity. As a result, our model better describes shadows, light reflections, and the transparency of 3D objects without adding additional texture and light components.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.13729v5.pdf)

### GaussianVolumes
#### GVGEN: Text-to-3D Generation with Volumetric Representation

**Authors**: Tong He, Wanli Ouyang, Chun Yuan, Xiaoshui Huang, Yangguang Li, Di Huang, Sida Peng, Junyi Chen, Xianglong He

<details>
<summary><b>Abstract</b></summary>
In recent years, 3D Gaussian splatting has emerged as a powerful technique for 3D reconstruction and generation, known for its fast and high-quality rendering capabilities. To address these shortcomings, this paper introduces a novel diffusion-based framework, GVGEN, designed to efficiently generate 3D Gaussian representations from text input. We propose two innovative techniques:(1) Structured Volumetric Representation. We first arrange disorganized 3D Gaussian points as a structured form GaussianVolume. This transformation allows the capture of intricate texture details within a volume composed of a fixed number of Gaussians. To better optimize the representation of these details, we propose a unique pruning and densifying method named the Candidate Pool Strategy, enhancing detail fidelity through selective optimization. (2) Coarse-to-fine Generation Pipeline. To simplify the generation of GaussianVolume and empower the model to generate instances with detailed 3D geometry, we propose a coarse-to-fine pipeline. It initially constructs a basic geometric structure, followed by the prediction of complete Gaussian attributes. Our framework, GVGEN, demonstrates superior performance in qualitative and quantitative assessments compared to existing 3D generation methods. Simultaneously, it maintains a fast generation speed ($\sim$7 seconds), effectively striking a balance between quality and efficiency. Our project page is: https://gvgen.github.io/
</details>

[📄 Paper](https://arxiv.org/pdf/2403.12957v2.pdf)

## Guidance by Additional Prior
#### Human Gaussian Splatting: Real-time Rendering of Animatable Avatars

**Authors**: Eduardo Pérez-Pellitero, Yiren Zhou, Richard Shaw, Helisa Dhamo, Jifei Song, Arthur Moreau

<details>
<summary><b>Abstract</b></summary>
This work addresses the problem of real-time rendering of photorealistic human body avatars learned from multi-view videos. While the classical approaches to model and render virtual humans generally use a textured mesh, recent research has developed neural body representations that achieve impressive visual quality. However, these models are difficult to render in real-time and their quality degrades when the character is animated with body poses different than the training observations. We propose an animatable human model based on 3D Gaussian Splatting, that has recently emerged as a very efficient alternative to neural radiance fields. The body is represented by a set of gaussian primitives in a canonical space which is deformed with a coarse to fine approach that combines forward skinning and local non-rigid refinement. We describe how to learn our Human Gaussian Splatting (HuGS) model in an end-to-end fashion from multi-view observations, and evaluate it against the state-of-the-art approaches for novel pose synthesis of clothed body. Our method achieves 1.5 dB PSNR improvement over the state-of-the-art on THuman4 dataset while being able to render in real-time (80 fps for 512x512 resolution).
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17113v2.pdf)

#### HUGS: Human Gaussian Splats

**Authors**: Anurag Ranjan, Oncel Tuzel, James Gabriel, Jen-Hao Rick Chang, Muhammed Kocabas

<details>
<summary><b>Abstract</b></summary>
Recent advances in neural rendering have improved both training and rendering times by orders of magnitude. While these methods demonstrate state-of-the-art quality and speed, they are designed for photogrammetry of static scenes and do not generalize well to freely moving humans in the environment. In this work, we introduce Human Gaussian Splats (HUGS) that represents an animatable human together with the scene using 3D Gaussian Splatting (3DGS). Our method takes only a monocular video with a small number of (50-100) frames, and it automatically learns to disentangle the static scene and a fully animatable human avatar within 30 minutes. We utilize the SMPL body model to initialize the human Gaussians. To capture details that are not modeled by SMPL (e.g. cloth, hairs), we allow the 3D Gaussians to deviate from the human body model. Utilizing 3D Gaussians for animated humans brings new challenges, including the artifacts created when articulating the Gaussians. We propose to jointly optimize the linear blend skinning weights to coordinate the movements of individual Gaussians during animation. Our approach enables novel-pose synthesis of human and novel view synthesis of both the human and the scene. We achieve state-of-the-art rendering quality with a rendering speed of 60 FPS while being ~100x faster to train over previous work. Our code will be announced here: https://github.com/apple/ml-hugs
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17910v1.pdf)

#### 3D Menagerie: Modeling the 3D shape and pose of animals

**Authors**: Michael J. Black, Silvia Zuffi, Angjoo Kanazawa, David Jacobs

<details>
<summary><b>Abstract</b></summary>
There has been significant work on learning realistic, articulated, 3D models
of the human body. In contrast, there are few such models of animals, despite
many applications. The main challenge is that animals are much less cooperative
than humans. The best human body models are learned from thousands of 3D scans
of people in specific poses, which is infeasible with live animals.
Consequently, we learn our model from a small set of 3D scans of toy figurines
in arbitrary poses. We employ a novel part-based shape model to compute an
initial registration to the scans. We then normalize their pose, learn a
statistical shape model, and refine the registrations and the model together.
In this way, we accurately align animal scans from different quadruped families
with very different shapes and poses. With the registration to a common
template we learn a shape space representing animals including lions, cats,
dogs, horses, cows and hippos. Animal shapes can be sampled from the model,
posed, animated, and fit to data. We demonstrate generalization by fitting it
to images of real animals including species not seen in training.
</details>

[📄 Paper](http://arxiv.org/pdf/1611.07700v2.pdf)

#### 3DGS-Avatar: Animatable Avatars via Deformable 3D Gaussian Splatting

**Authors**: Siyu Tang, Andreas Geiger, Marko Mihajlovic, Shaofei Wang, Zhiyin Qian

<details>
<summary><b>Abstract</b></summary>
We introduce an approach that creates animatable human avatars from monocular videos using 3D Gaussian Splatting (3DGS). Existing methods based on neural radiance fields (NeRFs) achieve high-quality novel-view/novel-pose image synthesis but often require days of training, and are extremely slow at inference time. Recently, the community has explored fast grid structures for efficient training of clothed avatars. Albeit being extremely fast at training, these methods can barely achieve an interactive rendering frame rate with around 15 FPS. In this paper, we use 3D Gaussian Splatting and learn a non-rigid deformation network to reconstruct animatable clothed human avatars that can be trained within 30 minutes and rendered at real-time frame rates (50+ FPS). Given the explicit nature of our representation, we further introduce as-isometric-as-possible regularizations on both the Gaussian mean vectors and the covariance matrices, enhancing the generalization of our model on highly articulated unseen poses. Experimental results show that our method achieves comparable and even better performance compared to state-of-the-art approaches on animatable avatar creation from a monocular input, while being 400x and 250x faster in training and inference, respectively.
</details>
[📄 Paper](https://arxiv.org/pdf/2312.09228v3.pdf)

#### Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting

**Authors**: Monroe Kennedy III, Mac Schwager, Gadiel Sznaier Camps, Won Kyung Do, Matthew Strong, Aiden Swann

<details>
<summary><b>Abstract</b></summary>
In this work, we propose a novel method to supervise 3D Gaussian Splatting (3DGS) scenes using optical tactile sensors. Optical tactile sensors have become widespread in their use in robotics for manipulation and object representation; however, raw optical tactile sensor data is unsuitable to directly supervise a 3DGS scene. Our representation leverages a Gaussian Process Implicit Surface to implicitly represent the object, combining many touches into a unified representation with uncertainty. We merge this model with a monocular depth estimation network, which is aligned in a two stage process, coarsely aligning with a depth camera and then finely adjusting to match our touch data. For every training image, our method produces a corresponding fused depth and uncertainty map. Utilizing this additional information, we propose a new loss function, variance weighted depth supervised loss, for training the 3DGS scene model. We leverage the DenseTact optical tactile sensor and RealSense RGB-D camera to show that combining touch and vision in this manner leads to quantitatively and qualitatively better results than vision or touch alone in a few-view scene syntheses on opaque as well as on reflective and transparent objects. Please see our project page at http://armlabstanford.github.io/touch-gs
</details>

[📄 Paper](https://arxiv.org/pdf/2403.09875v2.pdf)

#### GauHuman: Articulated Gaussian Splatting from Monocular Human Videos

**Authors**: Ziwei Liu, Shoukang Hu

<details>
<summary><b>Abstract</b></summary>
We present, GauHuman, a 3D human model with Gaussian Splatting for both fast training (1 ~ 2 minutes) and real-time rendering (up to 189 FPS), compared with existing NeRF-based implicit representation modelling frameworks demanding hours of training and seconds of rendering per frame. Specifically, GauHuman encodes Gaussian Splatting in the canonical space and transforms 3D Gaussians from canonical space to posed space with linear blend skinning (LBS), in which effective pose and LBS refinement modules are designed to learn fine details of 3D humans under negligible computational cost. Moreover, to enable fast optimization of GauHuman, we initialize and prune 3D Gaussians with 3D human prior, while splitting/cloning via KL divergence guidance, along with a novel merge operation for further speeding up. Extensive experiments on ZJU_Mocap and MonoCap datasets demonstrate that GauHuman achieves state-of-the-art performance quantitatively and qualitatively with fast training and real-time rendering speed. Notably, without sacrificing rendering quality, GauHuman can fast model the 3D human performer with ~13k 3D Gaussians.
</details>

[📄 Paper](https://arxiv.org/pdf/2312.02973v1.pdf)

#### HumanGaussian: Text-Driven 3D Human Generation with Gaussian Splatting

**Authors**: Ziwei Liu, Xihui Liu, Dahua Lin, Gang Zeng, Ying Shan, Jiaxiang Tang, Xiaohang Zhan, Xian Liu

<details>
<summary><b>Abstract</b></summary>
Realistic 3D human generation from text prompts is a desirable yet challenging task. Existing methods optimize 3D representations like mesh or neural fields via score distillation sampling (SDS), which suffers from inadequate fine details or excessive training time. In this paper, we propose an efficient yet effective framework, HumanGaussian, that generates high-quality 3D humans with fine-grained geometry and realistic appearance. Our key insight is that 3D Gaussian Splatting is an efficient renderer with periodic Gaussian shrinkage or growing, where such adaptive density control can be naturally guided by intrinsic human structures. Specifically, 1) we first propose a Structure-Aware SDS that simultaneously optimizes human appearance and geometry. The multi-modal score function from both RGB and depth space is leveraged to distill the Gaussian densification and pruning process. 2) Moreover, we devise an Annealed Negative Prompt Guidance by decomposing SDS into a noisier generative score and a cleaner classifier score, which well addresses the over-saturation issue. The floating artifacts are further eliminated based on Gaussian size in a prune-only phase to enhance generation smoothness. Extensive experiments demonstrate the superior efficiency and competitive quality of our framework, rendering vivid 3D humans under diverse scenarios. Project Page: https://alvinliu0.github.io/projects/HumanGaussian
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17061v2.pdf)

#### Gaussian Shell Maps for Efficient 3D Human Generation

**Authors**: Gordon Wetzstein, Dit-yan Yeung, Qifeng Chen, Zhengfei Kuang, Ryan Po, Yinghao Xu, Zifan Shi, Wang Yifan, Rameen Abdal

<details>
<summary><b>Abstract</b></summary>
Efficient generation of 3D digital humans is important in several industries, including virtual reality, social media, and cinematic production. 3D generative adversarial networks (GANs) have demonstrated state-of-the-art (SOTA) quality and diversity for generated assets. Current 3D GAN architectures, however, typically rely on volume representations, which are slow to render, thereby hampering the GAN training and requiring multi-view-inconsistent 2D upsamplers. Here, we introduce Gaussian Shell Maps (GSMs) as a framework that connects SOTA generator network architectures with emerging 3D Gaussian rendering primitives using an articulable multi shell--based scaffold. In this setting, a CNN generates a 3D texture stack with features that are mapped to the shells. The latter represent inflated and deflated versions of a template surface of a digital human in a canonical body pose. Instead of rasterizing the shells directly, we sample 3D Gaussians on the shells whose attributes are encoded in the texture features. These Gaussians are efficiently and differentiably rendered. The ability to articulate the shells is important during GAN training and, at inference time, to deform a body into arbitrary user-defined poses. Our efficient rendering scheme bypasses the need for view-inconsistent upsamplers and achieves high-quality multi-view consistent renderings at a native resolution of $512 \times 512$ pixels. We demonstrate that GSMs successfully generate 3D humans when trained on single-view datasets, including SHHQ and DeepFashion.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17857v1.pdf)

#### GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting

**Authors**: Ming-Hsuan Yang, Deqing Sun, Yongtao Wang, Zhiwei Lin, Jinlin He, Yajiao Xiong, Xingjian Ran, Xiaoyu Zhou

<details>
<summary><b>Abstract</b></summary>
We present GALA3D, generative 3D GAussians with LAyout-guided control, for effective compositional text-to-3D generation. We first utilize large language models (LLMs) to generate the initial layout and introduce a layout-guided 3D Gaussian representation for 3D content generation with adaptive geometric constraints. We then propose an instance-scene compositional optimization mechanism with conditioned diffusion to collaboratively generate realistic 3D scenes with consistent geometry, texture, scale, and accurate interactions among multiple objects while simultaneously adjusting the coarse layout priors extracted from the LLMs to align with the generated scene. Experiments show that GALA3D is a user-friendly, end-to-end framework for state-of-the-art scene-level 3D content generation and controllable editing while ensuring the high fidelity of object-level entities within the scene. The source codes and models will be available at gala3d.github.io.
</details>

[📄 Paper](https://arxiv.org/pdf/2402.07207v2.pdf)

#### Gaussian Shell Maps for Efficient 3D Human Generation

**Authors**: Gordon Wetzstein, Dit-yan Yeung, Qifeng Chen, Zhengfei Kuang, Ryan Po, Yinghao Xu, Zifan Shi, Wang Yifan, Rameen Abdal

<details>
<summary><b>Abstract</b></summary>
Efficient generation of 3D digital humans is important in several industries, including virtual reality, social media, and cinematic production. 3D generative adversarial networks (GANs) have demonstrated state-of-the-art (SOTA) quality and diversity for generated assets. Current 3D GAN architectures, however, typically rely on volume representations, which are slow to render, thereby hampering the GAN training and requiring multi-view-inconsistent 2D upsamplers. Here, we introduce Gaussian Shell Maps (GSMs) as a framework that connects SOTA generator network architectures with emerging 3D Gaussian rendering primitives using an articulable multi shell--based scaffold. In this setting, a CNN generates a 3D texture stack with features that are mapped to the shells. The latter represent inflated and deflated versions of a template surface of a digital human in a canonical body pose. Instead of rasterizing the shells directly, we sample 3D Gaussians on the shells whose attributes are encoded in the texture features. These Gaussians are efficiently and differentiably rendered. The ability to articulate the shells is important during GAN training and, at inference time, to deform a body into arbitrary user-defined poses. Our efficient rendering scheme bypasses the need for view-inconsistent upsamplers and achieves high-quality multi-view consistent renderings at a native resolution of $512 \times 512$ pixels. We demonstrate that GSMs successfully generate 3D humans when trained on single-view datasets, including SHHQ and DeepFashion.
</details>

[📄 Paper](https://arxiv.org/pdf/2311.17857v1.pdf)

#### HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting

**Authors**: Yiyi Liao, Andreas Geiger, Yue Wang, Bingbing Liu, Weichao Qiu, Dongfeng Bai, Lu Xu, Jiahao Shao, HongYu Zhou

<details>
<summary><b>Abstract</b></summary>
Holistic understanding of urban scenes based on RGB images is a challenging yet important problem. It encompasses understanding both the geometry and appearance to enable novel view synthesis, parsing semantic labels, and tracking moving objects. Despite considerable progress, existing approaches often focus on specific aspects of this task and require additional inputs such as LiDAR scans or manually annotated 3D bounding boxes. In this paper, we introduce a novel pipeline that utilizes 3D Gaussian Splatting for holistic urban scene understanding. Our main idea involves the joint optimization of geometry, appearance, semantics, and motion using a combination of static and dynamic 3D Gaussians, where moving object poses are regularized via physical constraints. Our approach offers the ability to render new viewpoints in real-time, yielding 2D and 3D semantic information with high accuracy, and reconstruct dynamic scenes, even in scenarios where 3D bounding box detection are highly noisy. Experimental results on KITTI, KITTI-360, and Virtual KITTI 2 demonstrate the effectiveness of our approach.
</details>
[📄 Paper](https://arxiv.org/pdf/2403.12722v1.pdf)
