DeepCache: Accelerating Diffusion Models for Free

Xinyin Ma Gongfan Fang Xinchao Wang*
National University of Singapore
{maxinyin, gongfan}@u.nus.edu, xinchao@nus.edu.sg

3
2
0
2
c
e
D
7

]

V
C
.
s
c
[

2
v
8
5
8
0
0
.
2
1
3
2
:
v
i
X
r
a

Figure 1. Accelerating Stable Diffusion V1.5 and LDM-4-G by 2.3× and 7.0×, with 50 PLMS steps and 250 DDIM steps respectively.

Abstract

Diffusion models have recently gained unprecedented
attention in the field of image synthesis due to their re-
markable generative capabilities. Notwithstanding their
prowess,
these models often incur substantial computa-
tional costs, primarily attributed to the sequential denoising
process and cumbersome model size. Traditional methods
for compressing diffusion models typically involve extensive
retraining, presenting cost and feasibility challenges.
In
this paper, we introduce DeepCache, a novel training-free
paradigm that accelerates diffusion models from the per-
spective of model architecture. DeepCache capitalizes on
the inherent temporal redundancy observed in the sequen-
tial denoising steps of diffusion models, which caches and
retrieves features across adjacent denoising stages, thereby
curtailing redundant computations. Utilizing the property
of the U-Net, we reuse the high-level features while up-

dating the low-level features in a very cheap way. This
innovative strategy, in turn, enables a speedup factor of
2.3× for Stable Diffusion v1.5 with only a 0.05 decline
in CLIP Score, and 4.1× for LDM-4-G with a slight de-
crease of 0.22 in FID on ImageNet. Our experiments also
demonstrate DeepCache’s superiority over existing prun-
ing and distillation methods that necessitate retraining and
its compatibility with current sampling techniques. Fur-
thermore, we find that under the same throughput, Deep-
Cache effectively achieves comparable or even marginally
improved results with DDIM or PLMS. Code is available at
https://github.com/horseee/DeepCache.

1. Introduction

In recent years, diffusion models [9, 17, 57, 59] have
emerged as a pivotal advancement in the field of genera-

∗ Corresponding author

1

2.3 ×7.0 ×(a) Stable Diffusion v1.5(b) LDM-4-G for ImageNet

tive modeling, gaining substantial attention for their impres-
sive capabilities. These models have demonstrated remark-
able efficacy across diverse applications, being employed
for the generation of images [19, 60, 64], text [11, 28], au-
dio [6, 44], and video [18, 36, 56]. A large number of attrac-
tive applications have been facilitated with diffusion mod-
els, including but not limited to image editing [2, 20, 38],
image super-enhancing [26, 51], image-to-image transla-
tion [7, 49], text-to-image generation [41, 45, 46, 50] and
text-to-3D generation [30, 35, 43].

Despite the significant effectiveness of diffusion mod-
els, their relatively slow inference speed remains a ma-
jor obstacle to broader adoption, as highlighted in [29].
The core challenge stems from the step-by-step denois-
ing process required during their reverse phase, limiting
parallel decoding capabilities [55]. Efforts to accelerate
these models have focused on two main strategies: reduc-
ing the number of sampling steps, as seen in approaches
[34, 39, 52, 58], and decreasing the model inference over-
head per step through methods like model pruning, distilla-
tion, and quantization [10, 13, 21].

Our goal is to enhance the efficiency of diffusion models
by reducing model size at each step. Previous compres-
sion methods for diffusion models focused on re-designing
network architectures through a comprehensive structural
analysis [29] or involving frequency priors into the model
design [66], which yields promising results on image gen-
eration. However, they require large-scale datasets for re-
training these lightweight models. Pruning-based meth-
ods, as explored by [10, 21], lessen the data and training
requirements to 0.1% of the full training. Alternatively,
[32] employs adaptive models for different steps, which is
also a potential solution. However, it depends on a collec-
tion of pre-trained models or requires optimization of sub-
networks [65]. Those methods can reduce the expense of
crafting a new lightweight model, but the retraining process
is still inevitable, which makes the compression costly and
less practical for large-scale pre-trained diffusion models,
such as Stable Diffusion [47].

To this end, we focus on a challenging topic: How to
significantly reduce the computational overhead at each de-
noising step without additional training, thereby achieving
a cost-free compression of Diffusion Models? To achieve
this, we turn our attention to the intrinsic characteristics
of the reverse denoising process of diffusion models, ob-
serving a significant temporal consistency in the high-level
features between consecutive steps. We found that those
high-level features are even cacheable, which can be calcu-
lated once and retrieved again for the subsequent steps. By
leveraging the structural property of U-Net, the high-level
features can be cached while maintaining the low-level fea-
tures updated at each denoising step. Through this, a con-
siderable enhancement in the efficiency and speed of Diffu-

sion Models can be achieved without any training.

To summarize, we introduce a novel paradigm for the
acceleration of Diffusion Models, which gives a new per-
spective for training-free accelerating the diffusion models.
It is not merely compatible with existing fast samplers but
also shows potential for comparable or superior generation
capabilities. The contributions of our paper include:
• We introduce a simple and effective acceleration algo-
rithm, named DeepCache, for dynamically compressing
diffusion models during runtime and thus enhancing im-
age generation speed without additional training burdens.
• DeepCache utilizes the temporal consistency between
high-level features. With the cacheable features, the re-
dundant calculations are effectively reduced. Further-
more, we introduce a non-uniform 1:N strategy, specifi-
cally tailored for long caching intervals.

• DeepCache is validated across a variety of datasets, in-
cluding CIFAR, LSUN-Bedroom/Churches, ImageNet,
COCO2017 and PartiPrompt, and tested under DDPM,
LDM, and Stable Diffusion. Experiments demonstrate
that our approach has superior efficacy than pruning and
distillation algorithms that require retraining under the
same throughput.

2. Related Work

High-dimensional image generation has evolved signifi-
cantly in generative modeling.
Initially, GANs [1, 12]
and VAEs [16, 22] led this field but faced scalability is-
sues due to instability and mode collapse [23]. Recent ad-
vancements have been led by Diffusion Probabilistic Mod-
els [9, 17, 47, 61], which offer superior image generation.
However, the inherent nature of the reverse diffusion pro-
cess [59] slows down inference. Current research is focused
on two main methods to speed up diffusion model inference.

Optimized Sampling Efficiency.
focuses on reducing the
number of sampling steps. DDIM [58] reduces these
steps by exploring a non-Markovian process, related to neu-
ral ODEs. Studies
[3, 33, 34, 69] further dive into the
fast solver of SDE or ODE to create efficient sampling of
diffusion models. Some methods progressively distilled
the model to reduced timestep [52] or replace the remain-
ing steps with a single-step VAE [37]. The Consistency
Model [62] converts random noise to the initial images with
only one model evaluation. Parallel sampling techniques
like DSNO [70] and ParaDiGMS [55] employ Fourier neu-
ral operators and Picard iterations for parallel decoding .

Optimized Structural Efficiency. This approach aims to
reduce inference time at each sampling step. It leverages
strategies like structural pruning in Diff-pruning [10] and
efficient structure evolving in SnapFusion [29]. Spectral

2

Diffusion [66] enhances architectural design by incorpo-
rating frequency dynamics and priors. In contrast to these
methods, which use a uniform model at each step, [32] pro-
poses utilizing different models at various steps, selected
from a diffusion model zoo. The early stopping mechanism
in diffusion is explored in [27, 40, 63], while the quantiza-
tion techniques [13, 54] focus on low-precision data types
for model weights and activations. Additionally, [4] and
[5] present novel approaches to concentrate on inputs, with
the former adopting a unique forward process for each pixel
and the latter merging tokens based on similarity to enhance
computational efficiency in attention modules. Our method
is categorized under an objective to minimize the average
inference time per step. Uniquely, our approach reduces the
average model size substantially for each step, accelerating
the denoising process without necessitating retraining.

3. Methodology

3.1. Preliminary

Forward and Reverse Process. Diffusion models [17]
simulate an image generation process using a series of ran-
dom diffusion steps. The core idea behind diffusion models
is to start from random noise and gradually refine it until it
resembles a sample from the target distribution. In the for-
ward diffusion process, with a data point sampled from the
real distribution, x0 ∼ q(x), gaussian noises are gradually
added in T steps:

q (xt|xt−1) = N

(cid:16)
xt; (cid:112)1 − βtxt−1, βtI

(cid:17)

(1)

where t is the current timestep and {βt} schedules the noise.
The reverse diffusion process denoises the random noise
xT ∼ N (0, I) into the target distribution by modeling
q (xt−1|xt). At each reverse step t, the conditional prob-
ability distribution is approximated by a network ϵθ (xt, t)
with the timestep t and previous output xt as input:

xt−1 ∼ pθ(xt−1|xt) =

(cid:18)

N

xt−1;

(cid:18)

xt −

1
√
αt

1 − αt
√
1 − ¯αt

(cid:19)

(cid:19)

ϵθ (xt, t)

, βtI

(2)

where αt = 1 − βt and ¯αt = (cid:81)T
i=1 αi. Applied iteratively,
it gradually reduces the noise of the current xt, bringing it
close to a real data point when we reach x0.

in U-Net. U-
High-level and Low-level Features
Net [48] was originally introduced for biomedical image
segmentation and showcased a strong ability to amalgamate
low-level and high-level features, attributed to the skip con-
nections. U-Net is constructed on stacked downsampling
and upsampling blocks, which encode the input image into a
high-level representation and then decode it for downstream

i=1 and {Ui}d

tasks. The block pairs, denoted as {Di}d
i=1,
are interconnected with additional skip paths. Those skip
paths directly forward the rich and relatively more low-level
information from Di to Ui. During the forward propaga-
tion in the U-Net architecture, the data traverses concur-
rently through two pathways: the main branch and the skip
branch. These branches converge at a concatenation mod-
ule, with the main branch providing processed high-level
feature from the preceding upsampling block Ui+1, and
the skip branch contributing corresponding feature from the
symmetrical block Di. Therefore, at the heart of a U-Net
model is a concatenation of low-level features from the skip
branch, and the high-level features from the main branch,
formalized as:

Concat(Di(·), Ui+1(·))

(3)

3.2. Feature Redundancy in Sequential Denoising

The inherent sequentiality of the denoising process in dif-
fusion models presents a primary bottleneck in inference
speed. Previous methods primarily employed strategies that
involved skipping certain steps to address this issue. In this
work, we revisit the entire denoising process, seeking to un-
cover specific properties that could be optimized to enhance
inference efficiency.

Observation. Adjacent steps in the denoising process ex-
hibit significant temporal similarity in high-level features.

In Figure 2, we provide empirical evidence related to
this observation. The experiments elucidate two primary
insights: 1) There is a noticeable temporal feature similar-
ity between adjacent steps within the denoising process, in-
dicating that the change between consecutive steps is typi-
cally minor; 2) Regardless of the diffusion model we used,
for each timestep, at least 10% of the adjacent timesteps
exhibit a high similarity (>0.95) to the current step, sug-
gesting that certain high-level features change at a gradual
pace. This phenomenon can be observed in a large num-
ber of well-established models like Stable Diffusion, LDM,
and DDPM. In the case of DDPM for LSUN-Churches and
LSUN-Bedroom, some timesteps even demonstrate a high
degree of similarity to 80% of the other steps, as highlighted
in the green line in Figure 2 (c).

Building upon these observations, our objective is to
leverage this advantageous characteristic to accelerate the
denoising process. Our analysis reveals that the compu-
tation often results in a feature remarkably similar to that
of the previous step, thereby highlighting the existence of
redundant calculations for optimization. We contend that
allocating significant computational resources to regener-
ate these analogous feature maps constitutes an inefficiency.
While incurring substantial computational expense, yields
marginal benefits, it suggests a potential area for efficiency
improvements in the speed of diffusion models.

3

Figure 2. (a) Examples of feature maps in the up-sampling block U2 in Stable Diffusion. We present a comparison from two adjacently
paired steps, emphasizing the invariance inherent in the denoising process. (b) Heatmap of similarity between U2’s features in all steps on
three typical diffusion models. (c) The percentage of steps with a similarity greater than 0.95 to the current step.

3.3. Deep Cache For Diffusion Models

We introduce DeepCache, a simple and effective approach
that leverages the temporal redundancy between steps in
the reverse diffusion process to accelerate inference. Our
method draws inspiration from the caching mechanism in
a computer system, incorporating a storage component de-
signed for elements that exhibit minimal changes over time.
Applying this in diffusion models, we eliminate redundant
computations by strategically caching slowly evolving fea-
tures, thereby obviating the need for repetitive recalcula-
tions in subsequent steps.

To achieve this, we shift our focus to the skip connec-
tions within U-Net, which inherently offers a dual-pathway
the main branch requires heavy computation
advantage:
to traverse the entire network, while the skip branch only
needs to go through some shallow layers, resulting in a very
small computational load. The prominent feature similarity
in the main branch, allows us to reuse the already computed
results rather than calculate it repeatedly for all timesteps.

Cacheable Features in denosing. To make this idea
more concrete, we study the case within two consecutive
timesteps t and t − 1. According to the reverse process,
xt−1 would be conditionally generated based on the pre-
vious results xt. First, we generate xt in the same way
as usual, where the calculations are performed across the
entire U-Net. To obtain the next output xt−1, we retrieve
the high-level features produced in the previous xt. More
specifically, consider a skip branch m in the U-Net, which
bridges Dm and Um, we cache the feature maps from the
previous up-sampling block at the time t as the following:
cache ← U t
F t

m+1(·)

(4)

which is the feature from the main branch at timestep t.
Those cached features will be plugged into the network in-
ference in the subsequent steps. In the next timestep t − 1,
the inference is not carried out on the entire network; in-
stead, we perform a dynamic partial inference. Based on
the previously generated xt, we only calculate those that are
necessary for the m-th skip branch and substitute the com-
pute of the main branch with a retrieving operation from
the cache in Equation 4. Therefore, the input for U t−1
m in
the t − 1 timestep can be formulated as:

Concat(Dt−1

m (·), F t

cache)

(5)

Here, Dt−1
m represents the output of the m-th down-
sampling block, which only contains a few layers if a small
m is selected. For example, if we perform DeepCache at
the first layer with m = 1, then we only need to execute
one downsampling block to obtain Dt−1
. As for the second
feature F t
cache, no additional computational cost is needed
since it can be simply retrieved from the cache. We illus-
trate the above process in Figure 3.

1

Extending to 1:N Inference This process is not limited
to the type with one step of full inference followed by one
step of partial inference. As shown in Figure 2(b), pair-
wise similarity remains a high value in several consecutive
steps. The mechanism can be extended to cover more steps,
with the cached features calculated once and reused in the
consecutive N − 1 steps to replace the original U t−n
m+1(·),
n ∈ {1, . . . , N − 1}. Thus, for all the T steps for denoising,
the sequence of timesteps that performs full inference are:

I = {x ∈ N | x = iN, 0 ≤ i < k}

(6)

where k = ⌈T /N ⌉ denotes the times for cache updating.

4

A large teddy bear with a heart is in the garbageA green plate filled with rice and a mixture of sauce on top of itA very ornate, three layeredwedding cake in a banquet roomPromptLDM-4-GDDPM/BedroomSD v1.5(c)Ratio of steps that similarity larger than 0.95Step1Step20Step19Step0Original(a) Examples of Feature Maps(b) HeatMapfor SimilarityFigure 3. Illustration of DeepCache. At the t−1 step, xt−1 is gen-
erated by reusing the features cached at the t step, and the blocks
D2, D3, U2, U3 are not executed for more efficient inference.

Non-uniform 1:N Inference Based on the 1:N strategy,
we managed to accelerate the inference of diffusion under
a strong assumption that the high-level features are invari-
ant in the consecutive N step. However, it’s not invariably
the case, especially for a large N, as demonstrated by the
experimental evidence in Figure 2(c). The similarity of the
features does not remain constant across all steps. For mod-
els such as LDM, the temporal similarity of features would
significantly decrease around 40% of the denoising process.
Thus, for the non-uniform 1:N inference, we tend to sample
more on those steps with relatively small similarities to the
adjacent steps. Here, the sequence of timesteps to perform
full inference becomes:

(cid:110)

li | li ∈ linear space

L =
I = unique int ({ik | ik = (lk)p + c, where lk ∈ L})

(−c)

1
p , (T − c)

1
p , k

(7)

(cid:17)(cid:111)

(cid:16)

where linear space(s, e, n) evenly spaces n numbers from
s to e (exclusive) and unique int(·) convert the number to
int and ensure the uniqueness of timesteps in the sequence.
c is the hyper-parameter for the selected center timestep.
In this equation, the frequency of indexes changes in a
quadratic manner as it moves away from a central timestep.
It is essential to note that the aforementioned strategy rep-
resents one among several potential strategies. Alternative
sequences, particularly centered on a specific timestep, can
also yield similar improvements in image quality.

4. Experiment

4.1. Experimental Settings

Models, Datasets and Evaluation Metrics To demon-
strate the effectiveness of our method is agnostic with the
type of pre-trained DMs, we evaluate our methods on three
commonly used DMs: DDPM [17], LDM [47] and Stable

5

Figure 4. MACs for each skip branch, evaluated on DDPM for
CIFAR10 and Stable Diffusion V1.5.

Diffusion [47]1. Except for this, to show that our method
is compatible with the fast sampling methods, we build our
method upon 100-step DDIM [58] for DDPM, 250-step for
LDM and 50-step PLMS [33] for Stable Diffusion, instead
of the complete 1000-step denoising process. We select six
datasets that are commonly adopted to evaluate these mod-
els, including CIFAR10 [24], LSUN-Bedroom [67], LSUN-
Churches [67], ImageNet [8], MS-COCO 2017 [31] and
PartiPrompts [68]. For MS-COCO 2017 and PartiPrompt,
we utilized the 5k validation set and 1.63k captions respec-
tively as prompts for Stable Diffusion. For other datasets,
we generate 50k images to assess the generation quality. We
follow previous works [10, 55, 66] to employ the evaluation
metrics including FID, sFID, IS, Precision-and-Recall and
CLIP Score (on ViT-g/14) [14, 15, 25, 53].

Baselines We choose Diff-Pruning [10] as the main base-
line for our method since Diff-Pruning also reduces the
training effort for compressing DMs. For the experiment
on LDM, we extend [66] as another baseline to represent
one of the best methods to re-design a lightweight diffusion
model. For the experiment on Stable Diffusion, we choose
BK-SDMs [21], which are trained on 2.3M LAION image-
text pairs, as baselines of architecturally compression and
distillation for Stable Diffusion.

4.2. Complexity Analysis

We first analyze the improvement in inference speed facil-
itated by DeepCache. The notable acceleration in infer-
ence speed primarily arises from incomplete reasoning in
denoising steps, with layer removal accomplished by parti-
tioning the U-Net by the skip connection. In Figure 4, we
present the division of MACs on two models. For each skip
branch i, the MACs here contain the MACs in down block
Di and the up block Ui. There is a difference in the amount

1https://huggingface.co/runwayml/stable-diffusion-v1-5

!!!"!#M#!#"##Skip Branch 1#!!!!!"$#%&%'(!"$"#"'###'$ChosenPathSkip Branch 2Skip Branch 3ExcludedPath!!!!)$(a)DDPMforCIFAR10(b) Stable Diffusion v1.5Method

IDDPM [42]
ADM-G [9]
LDM-4 [47]
LDM-4*

Spectral DPM [66]
Diff-Pruning [10]*

Uniform - N=2
Uniform - N=3
Uniform - N=5
Uniform - N=10
Uniform - N=20

NonUniform - N=10
NonUniform - N=20

MACs ↓ Throughput ↑

Speed ↑ Retrain

FID ↓

sFID ↓

IS ↑

Precision ↑ Recall ↑

ImageNet 256 × 256

1416.3G
1186.4G
99.82G
99.82G

9.9G
52.71G

52.12G
36.48G
23.50G
13.97G
9.39G

13.97G
9.39G

-
-
0.178
0.178

-
0.269

0.334
0.471
0.733
1.239
1.876

1.239
1.876

-
-
1×
1×

-
1.51×

1.88×
2.65×
4.12×
6.96×
10.54×

6.96×
10.54×

✗
✗
✗
✗

✓
✓

✗
✗
✗
✗
✗

✗
✗

12.26
4.59
3.60
3.37

5.42
5.25
-
5.14

-
186.70
247.67
204.56

10.60
9.27(9.16)

-
10.59

-
214.42(201.81)

3.39
3.44
3.59
4.41
8.23

4.27
7.11

5.11
5.11
5.16
5.57
8.08

5.42
7.34

204.09
202.79
200.45
191.11
161.83

193.11
167.85

70.0
82.0
87.0
82.71

-
87.87

82.75
82.65
82.36
81.26
75.31

81.75
77.44

62.0
52.0
48.0
53.86

-
30.87

54.07
53.81
53.31
51.53
50.57

51.84
50.08

Table 1. Class-conditional generation quality on ImageNet using LDM-4-G. The baselines here, as well as our methods, employ 250 DDIM
steps. *We reproduce Diff-Pruning to have a comprehensive comparison and the official results are shown in brackets.

Method

DDPM
DDPM*

Diff-Pruning

Ours - N=2
Ours - N=3
Ours - N=5
Ours - N=10

Method

DDPM
DDPM*

248.7G
248.7G

Diff-Pruning

138.8G

Ours - N=2
Ours - N=3
Ours - N=5

190.8G
172.3G
156.0G

Method

DDPM
DDPM*

248.7G
248.7G

Diff-Pruning

138.8G

Ours - N=2
Ours - N=3
Ours - N=5

190.8G
172.3G
156.0G

CIFAR-10 32 × 32

MACs ↓ Throughput ↑

Speed ↑

Steps ↓

FID ↓

6.1G
6.1G

3.4G

4.15G
3.54G
3.01G
2.63G

9.79
9.79

13.45

13.73
15.74
18.11
20.26

1×
1×

1.37×

1.40×
1.61×
1.85×
2.07×

-
-

100k

0
0
0
0

4.19
4.16

5.29

4.35
4.70
5.73
9.74

LSUN-Bedroom 256 × 256

MACs ↓ Throughput ↑

Speed ↑

Steps ↓

FID ↓

LSUN-Churches 256 × 256

MACs ↓ Throughput ↑

Speed ↑

Steps ↓

FID ↓

0.21
0.21

0.31

0.27
0.30
0.31

1×
1×

1.48×

1.29×
1.43×
1.48×

-
-

6.62
6.70

200k

18.60

0
0
0

6.69
7.20
9.49

0.21
0.21

0.31

0.27
0.30
0.31

1×
1×

1.48×

1.29×
1.43×
1.48×

-
-

500k

0
0
0

10.58
10.87

13.90

11.31
11.75
13.68

Table 2. Results on CIFAR-10, LSUN-Bedroom and LSUN-
Churches. All the methods here adopt 100 DDIM steps. * means
the reproduced results, which are more comparable with our re-
sults since the random seed is the same.

of computation allocated to different skip branches for dif-
ferent models. Stable diffusion demonstrates a compara-
tively uniform distribution across layers, whereas DDPM
exhibits more computational burden concentrated within the
first several layers. Our approach would benefit from U-Net

6

structures that have a larger number of skip branches, facili-
tating finer divisions of models, and giving us more choices
for trade-off the speed and quality. In our experiment, we
choose the skip branch 3/1/2 for DDPMs, LDM-4-G and
Stable Diffusion respectively. We provide the results of us-
ing different branches in Appendix.

To comprehensively evaluate the efficiency of our
method,
the
in the following experiments, we report
throughput of each model using a single RTX2080 GPU.
Besides, we report MACs in those tables, which refer to the
average MACs for all steps.

4.3. Comparison with Compression Methods

LDM-4-G for ImageNet. We conduct experiments on
ImageNet, and the results are shown in Table 1. When ac-
celerating to 4.1× the speed, a minor performance decline
is observed (from 3.39 to 3.59). Compared with the pruning
and distillation methods, a notable improvement over those
methods is observed in FID and sFID, even though the ac-
celeration ratio of our method is more substantial. Further-
more, the augmentation in quality becomes more obvious
with a larger number N of caching intervals if we employ
the non-uniform 1:N strategy. Detailed results for the non-
uniform 1:N strategy for small N and the hyper-parameters
for the non-uniform strategy are provided in the Appendix.

DDPMs for CIFAR-10 and LSUN. The results on CI-
FAR10, LSUN-Bedroom and LSUN-Churches are shown in
Table 2. From these tables, we can find out that our method
surpasses those requiring retraining, even though our meth-
ods have no retraining cost. Additionally, since we adopt
a layer-pruning approach, which is more hardware-friendly,
our acceleration ratio is more significant compared to the
baseline method, under similar MACs constraints.

Figure 5. Visualization of the generated images by BK-SDM-Tiny and DeepCache. All the methods adopt the 50-step PLMS. The time
here is the duration to generate a single image. Some prompts are omitted from this section for brevity. See Appendix for details.

Method

MACs ↓ Throughput ↑

Speed ↑ CLIP Score ↑ MACs ↓ Throughput ↑

Speed ↑ CLIP Score ↑

PartiPrompts

COCO2017

PLMS - 50 steps

338.83G

PLMS - 25 steps
BK-SDM - Base
BK-SDM - Small
BK-SDM - Tiny

Ours

169.42G
223.81G
217.78G
205.09G

130.45G

0.230

0.470
0.343
0.402
0.416

0.494

1.00×

2.04×
1.49×
1.75×
1.81×

2.15×

29.51

29.33
28.88
27.94
27.36

29.46

338.83G

169.42G
223.81G
217.78G
205.09G

130.45G

0.237

0.453
0.344
0.397
0.415

0.500

1.00×

1.91×
1.45 ×
1.68×
1.76 ×

2.11×

30.30

29.99
29.47
27.96
27.41

30.23

Table 3. Comparison with PLMS and BK-SDM. We utilized prompts in PartiPrompt and COCO2017 validation set to generate images
at the resolution of 512. We choose N=5 to achieve a throughput that is comparable to or surpasses that of established baseline methods.
Results for other choices of N can be found in Figure 6.

Stable Diffusion. The results are presented in Table 3.
We outperform all three variants of BK-SDM, even with a
faster denoising speed. As evident from the showcased ex-
amples in Figure 5, the images generated by our method ex-
hibit a greater consistency with the images generated by the
original diffusion model, and the image aligns better with
the textual prompt.

4.4. Comparison with Fast Sampler.

We conducted a comparative analysis with methods focused
on reducing sampling steps. It is essential to highlight that
our approach is additive to those fast samplers, as we show
in previous experiments. In Table 3 and Table 4, we first
compared our method with the DDIM [58] or PLMS [33]
under similar throughputs. We observe that our method
achieved slightly better results than 25-step PLMS on Sta-
ble Diffusion and comparable results to DDIM on LDM-4-
G. We also measured the performance comparison between
PLMS and our method under different acceleration ratios in
Figure 6 to provide a more comprehensive comparison.

Figure 6. Comparison between PLMS, DeepCache with uniform
1:N and non-uniform 1:N stratigies.

4.5. Analysis

Ablation Study. DeepCache can be conceptualized as in-
corporating (N −1)×K steps of shallow network inference
on top of the DDIM’s K steps, along with more updates of
the noisy images.
It is important to validate whether the
additional computations of shallow network inference and
the caching of features yield positive effectiveness: 1) Ef-
fectiveness of Cached Features: We assess the impact of

7

A person in a helmet is riding a skateboardThere are three vases made of clay on a tableA bicycle is standing next to a bed in a roomA kitten that is sitting down by a  doorA serene mountain landscape with a flowing river, lush greenery…A delicate floral arrangement with soft, pastel colors and light…A magical winter wonderland at night. Envision a landscape…A photograph of an abandoned house at the edge of a forest…A man holding a surfboard walking on a beach next to the oceanStable Diffusion v1.5BK-SDM-TinyOurs1.001.251.501.752.002.252.502.75Speedup Ratio28.628.829.029.229.429.6CLIP ScorePartiPromptPLMSUniform 1:NNon-Uniform 1:N1.01.21.41.61.82.02.22.42.6Speedup Ratio29.429.629.830.030.230.4CLIP ScoreMS-COCO 2017PLMSUniform 1:NNon-Uniform 1:NFigure 7. Illustration of the evolution in generated images with increasing caching interval N.

Method

Throughput ↑

FID ↓

sFID ↓

Model

Dataset

DeepCache w/o Cached Features

DDIM - 59 steps
Ours

DDIM - 91 steps
Ours

0.727
0.733

0.436
0.471

3.59
3.59

3.46
3.44

5.14
5.16

50.6
5.11

DDPM
LDM-4-G ImageNet

Cifar10

9.74
7.36

192.98
312.12

Table 5. Effectiveness of Cached Features. Under identical hyper-
parameters, we replace the cached features with a zero matrix.

Table 4. Comparison with DDIM under the same throughput. Here
we conduct class-conditional for ImageNet using LDM-4-G.

cached features in Table 5. Remarkably, we observe that,
without any retraining, the cached features play a pivotal
role in the effective denoising of diffusion models employ-
ing a shallow U-Net. 2) Positive Impact of Shallow Net-
work Inference: Building upon the cached features, the
shallow network inference we conduct has a positive im-
pact compared to DDIM. Results presented in Table 6 in-
dicate that, with the additional computation of the shallow
U-Net, DeepCache improves the 50-step DDIM by 0.32 and
the 10-step DDIM by 2.98.

Illustration of the increasing caching interval N.
In
Figure 7, we illustrate the evolution of generated images
as we increment the caching interval. A discernible trend
emerges as a gradual reduction in time, revealing that the
primary features of the images remain consistent with their
predecessors. However, subtle details such as the color of
clothing and the shape of the cat undergo modifications.
Quantitative insights are provided in Table 1 and Figure 6,
where with an interval N < 5, there is only a slight reduc-
tion in the quality of the generated image.

5. Limitations

The primary limitation of our method originates from its de-
pendence on the pre-defined structure of the pre-trained dif-
fusion model. Specifically, when a model’s shallowest skip
branch encompasses a substantial portion of computation,
such as 50% of the whole model, the achievable speedup

Steps DDIM FID↓ DeepCache FID↓

∆

50
20
10

4.67
6.84
13.36

4.35
5.73
10.38

-0.32
-1.11
-2.98

Table 6. Effectiveness of Shallow Network Inference. Steps here
mean the number of steps that perform full model inference.

ratio through our approach becomes relatively constrained.
Additionally, our method encounters non-negligible perfor-
mance degradation with larger caching steps (e.g., N=20),
which could impose constraints on the upper limit of the
acceleration ratio.

6. Conclusion

In this paper, we introduce a novel paradigm, DeepCache,
to accelerate the diffusion model. Our strategy employs the
similarity observed in high-level features across adjacent
steps of the diffusion model, thereby mitigating the compu-
tational overhead associated with redundant high-level fea-
ture calculations. Additionally, we leverage the structural
attributes in the U-Net architecture to facilitate the updat-
ing of low-level features. Through the adoption of Deep-
Cache, a noteworthy acceleration in computational speed is
achieved. Empirical evaluations on several datasets and dif-
fusion models demonstrate that DeepCache surpass other
compression methods that focus on the reduction of param-
eter size. Moreover, the proposed algorithm demonstrates
comparable and even slightly superior generation quality
compared to existing techniques such as DDIM and PLMS,
thereby offering a new perspective in the field.

8

A cat standing on the edge of a sink drink waterA child riding a skate boardon a city streetReferences

[1] Martin Arjovsky, Soumith Chintala, and L´eon Bottou.
In Interna-
Wasserstein generative adversarial networks.
tional conference on machine learning, pages 214–223.
PMLR, 2017. 2

[2] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended
diffusion for text-driven editing of natural images. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 18208–18218, 2022. 2
[3] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-
an analytic estimate of the optimal reverse vari-
arXiv preprint

dpm:
ance in diffusion probabilistic models.
arXiv:2201.06503, 2022. 2

[4] Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Sch¨onlieb,
and Christian Etmann. Non-uniform diffusion models. arXiv
preprint arXiv:2207.09786, 2022. 3

[5] Daniel Bolya and Judy Hoffman. Token merging for fast sta-
ble diffusion. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 4598–
4602, 2023. 3

[6] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J Weiss, Mo-
hammad Norouzi, and William Chan. Wavegrad: Esti-
mating gradients for waveform generation. arXiv preprint
arXiv:2009.00713, 2020. 2

[7] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune
Ilvr: Conditioning method for
arXiv preprint

Gwon, and Sungroh Yoon.
denoising diffusion probabilistic models.
arXiv:2108.02938, 2021. 2

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In 2009 IEEE conference on computer vision and
pattern recognition, pages 248–255. Ieee, 2009. 5

[9] Prafulla Dhariwal and Alexander Nichol. Diffusion models
beat gans on image synthesis. Advances in neural informa-
tion processing systems, 34:8780–8794, 2021. 1, 2, 6
[10] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural
pruning for diffusion models. In Advances in Neural Infor-
mation Processing Systems, 2023. 2, 5, 6

[11] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu,
and LingPeng Kong. Diffuseq: Sequence to sequence
arXiv preprint
text generation with diffusion models.
arXiv:2210.08933, 2022. 2

[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing
Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and
Yoshua Bengio. Generative adversarial nets. Advances in
neural information processing systems, 27, 2014. 2

[13] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou,
and Bohan Zhuang. Ptqd: Accurate post-training quantiza-
tion for diffusion models. arXiv preprint arXiv:2305.10657,
2023. 2, 3

[14] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras,
and Yejin Choi. Clipscore: A reference-free evaluation met-
ric for image captioning. arXiv preprint arXiv:2104.08718,
2021. 5

[15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
Bernhard Nessler, and Sepp Hochreiter. Gans trained by a

two time-scale update rule converge to a local nash equilib-
rium. Advances in neural information processing systems,
30, 2017. 5

[16] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess,
Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and
Alexander Lerchner. beta-vae: Learning basic visual con-
cepts with a constrained variational framework. In Interna-
tional conference on learning representations, 2016. 2
[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
fusion probabilistic models. Advances in neural information
processing systems, 33:6840–6851, 2020. 1, 2, 3, 5, 12
[18] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang,
Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben
Poole, Mohammad Norouzi, David J Fleet, et al.
Imagen
video: High definition video generation with diffusion mod-
els. arXiv preprint arXiv:2210.02303, 2022. 2

[19] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming
Song. Denoising diffusion restoration models. Advances
in Neural Information Processing Systems, 35:23593–23606,
2022. 2

[20] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen
Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic:
Text-based real image editing with diffusion models. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 6007–6017, 2023. 2

[21] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and
Shinkook Choi. On architectural compression of text-to-
image diffusion models. arXiv preprint arXiv:2305.15798,
2023. 2, 5

[22] Diederik P Kingma and Max Welling. Auto-encoding varia-
tional bayes. arXiv preprint arXiv:1312.6114, 2013. 2
[23] Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt
Kira. On convergence and stability of gans. arXiv preprint
arXiv:1705.07215, 2017. 2

[24] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple

layers of features from tiny images. 2009. 5

[25] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko
Lehtinen, and Timo Aila. Improved precision and recall met-
ric for assessing generative models. Advances in Neural In-
formation Processing Systems, 32, 2019. 5

[26] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun
Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single
image super-resolution with diffusion probabilistic models.
Neurocomputing, 479:47–59, 2022. 2

[27] Lijiang Li, Huixia Li, Xiawu Zheng, Jie Wu, Xuefeng Xiao,
Rui Wang, Min Zheng, Xin Pan, Fei Chao, and Rongrong Ji.
Autodiffusion: Training-free optimization of time steps and
architectures for automated diffusion model acceleration. In
Proceedings of the IEEE/CVF International Conference on
Computer Vision, pages 7105–7114, 2023. 3

[28] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang,
and Tatsunori B Hashimoto. Diffusion-lm improves control-
lable text generation. Advances in Neural Information Pro-
cessing Systems, 35:4328–4343, 2022. 2

[29] Yanyu Li, Huan Wang, Qing Jin, Ju Hu, Pavlo Chemerys,
Yun Fu, Yanzhi Wang, Sergey Tulyakov, and Jian Ren. Snap-
fusion: Text-to-image diffusion model on mobile devices

9

within two seconds. arXiv preprint arXiv:2306.00980, 2023.
2

[30] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa,
Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler,
Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution
text-to-3d content creation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 300–309, 2023. 2

[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays,
Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence
Zitnick. Microsoft coco: Common objects in context.
In
Computer Vision–ECCV 2014: 13th European Conference,
Zurich, Switzerland, September 6-12, 2014, Proceedings,
Part V 13, pages 740–755. Springer, 2014. 5

[32] Enshu Liu, Xuefei Ning, Zinan Lin, Huazhong Yang, and Yu
Wang. Oms-dpm: Optimizing the model schedule for diffu-
sion probabilistic models. arXiv preprint arXiv:2306.08860,
2023. 2, 3

[33] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo
numerical methods for diffusion models on manifolds. arXiv
preprint arXiv:2202.09778, 2022. 2, 5, 7

[34] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan
Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion
probabilistic model sampling in around 10 steps. Advances
in Neural Information Processing Systems, 35:5775–5787,
2022. 2

[35] Shitong Luo and Wei Hu. Diffusion probabilistic models for
3d point cloud generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 2837–2845, 2021. 2

[36] Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang,
Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and
Tieniu Tan. Videofusion: Decomposed diffusion mod-
In Proceedings of
els for high-quality video generation.
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10209–10218, 2023. 2

[37] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and
Bo Dai. Accelerating diffusion models via early stop of the
diffusion process. arXiv preprint arXiv:2205.12524, 2022. 2
[38] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jia-
jun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided
image synthesis and editing with stochastic differential equa-
tions. arXiv preprint arXiv:2108.01073, 2021. 2

[39] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik
Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans.
On distillation of guided diffusion models. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 14297–14306, 2023. 2

[40] Taehong Moon, Moonseok Choi, EungGu Yun, Jongmin
Yoon, Gayoung Lee, and Juho Lee. Early exiting for accel-
erated inference in diffusion models. In ICML 2023 Work-
shop on Structured Probabilistic Inference {\&} Generative
Modeling, 2023. 3

[41] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav
Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and
Mark Chen. Glide: Towards photorealistic image generation
and editing with text-guided diffusion models. arXiv preprint
arXiv:2112.10741, 2021. 2

[42] Alexander Quinn Nichol and Prafulla Dhariwal. Improved
In International
denoising diffusion probabilistic models.
Conference on Machine Learning, pages 8162–8171. PMLR,
2021. 6

[43] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv
preprint arXiv:2209.14988, 2022. 2

[44] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima
Sadekova, and Mikhail Kudinov. Grad-tts: A diffusion prob-
abilistic model for text-to-speech. In International Confer-
ence on Machine Learning, pages 8599–8608. PMLR, 2021.
2

[45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu,
and Mark Chen. Hierarchical text-conditional image gener-
ation with clip latents. arXiv preprint arXiv:2204.06125, 1
(2):3, 2022. 2

[46] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
In Proceedings of
synthesis with latent diffusion models.
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 2

[47] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
In Proceedings of
synthesis with latent diffusion models.
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 2, 5, 6

[48] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-
net: Convolutional networks for biomedical image segmen-
tation. In Medical Image Computing and Computer-Assisted
Intervention–MICCAI 2015: 18th International Conference,
Munich, Germany, October 5-9, 2015, Proceedings, Part III
18, pages 234–241. Springer, 2015. 3

[49] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee,
Jonathan Ho, Tim Salimans, David Fleet, and Mohammad
Norouzi. Palette:
In
ACM SIGGRAPH 2022 Conference Proceedings, pages 1–
10, 2022. 2

Image-to-image diffusion models.

[50] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with deep
language understanding. Advances in Neural Information
Processing Systems, 35:36479–36494, 2022. 2

[51] Chitwan Saharia, Jonathan Ho, William Chan, Tim Sal-
imans, David J Fleet, and Mohammad Norouzi.
Image
super-resolution via iterative refinement. IEEE Transactions
on Pattern Analysis and Machine Intelligence, 45(4):4713–
4726, 2022. 2

[52] Tim Salimans and Jonathan Ho. Progressive distillation
arXiv preprint

for fast sampling of diffusion models.
arXiv:2202.00512, 2022. 2

[53] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki
Cheung, Alec Radford, and Xi Chen. Improved techniques
for training gans. Advances in neural information processing
systems, 29, 2016. 5

[54] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and
Yan Yan. Post-training quantization on diffusion models. In

10

[70] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Aziz-
zadenesheli, and Anima Anandkumar. Fast sampling of dif-
fusion models via operator learning. In International Con-
ference on Machine Learning, pages 42390–42402. PMLR,
2023. 2

Proceedings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition, pages 1972–1981, 2023. 3
[55] Andy Shih, Suneel Belkhale, Stefano Ermon, Dorsa Sadigh,
and Nima Anari. Parallel sampling of diffusion models.
arXiv preprint arXiv:2305.16317, 2023. 2, 5

[56] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An,
Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual,
Oran Gafni, et al. Make-a-video: Text-to-video generation
without text-video data. arXiv preprint arXiv:2209.14792,
2022. 2

[57] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan,
and Surya Ganguli. Deep unsupervised learning using
In International confer-
nonequilibrium thermodynamics.
ence on machine learning, pages 2256–2265. PMLR, 2015.
1

[58] Jiaming Song, Chenlin Meng,

Denoising diffusion implicit models.
arXiv:2010.02502, 2020. 2, 5, 7

and Stefano Ermon.
arXiv preprint

[59] Yang Song and Stefano Ermon. Generative modeling by esti-
mating gradients of the data distribution. Advances in neural
information processing systems, 32, 2019. 1, 2

[60] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon.
Sliced score matching: A scalable approach to density and
In Uncertainty in Artificial Intelligence,
score estimation.
pages 574–584. PMLR, 2020. 2

[61] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
generative modeling through stochastic differential equa-
tions. arXiv preprint arXiv:2011.13456, 2020. 2

[62] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya

Sutskever. Consistency models. 2023. 2

[63] Shengkun Tang, Yaqing Wang, Caiwen Ding, Yi Liang, Yao
Li, and Dongkuan Xu. Deediff: Dynamic uncertainty-aware
early exiting for accelerating diffusion model generation.
arXiv preprint arXiv:2309.17074, 2023. 3

[64] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based
generative modeling in latent space. Advances in Neural In-
formation Processing Systems, 34:11287–11302, 2021. 2
[65] Shuai Yang, Yukang Chen, Luozhou Wang, Shu Liu, and
Yingcong Chen. Denoising diffusion step-aware models.
arXiv preprint arXiv:2310.03337, 2023. 2

[66] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang.
Diffusion probabilistic model made slim. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 22552–22562, 2023. 2, 3, 5, 6

[67] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas
Funkhouser, and Jianxiong Xiao. Lsun: Construction of a
large-scale image dataset using deep learning with humans
in the loop. arXiv preprint arXiv:1506.03365, 2015. 5
[68] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gun-
jan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yin-
fei Yang, Burcu Karagol Ayan, et al. Scaling autoregres-
sive models for content-rich text-to-image generation. arXiv
preprint arXiv:2206.10789, 2022. 5

[69] Qinsheng Zhang and Yongxin Chen. Fast sampling of dif-
fusion models with exponential integrator. arXiv preprint
arXiv:2204.13902, 2022. 2

11

DeepCache: Accelerating Diffusion Models for Free

Supplementary Material

A. Pseudo algorithm

We present the pseudocode for our algorithm in Algorithm
1. It illustrates the iterative generation process over N steps,
involving one step of complete model inference and N-1
steps of partial model inference. Here, we employ the sam-
pling algorithm from DDPM [17] as an example. Our algo-
rithm is adaptable to other fast sampling methods.

Algorithm 1: DeepCache

Input: A U-Net Model with down-sample blocks

i=1, up-sample blocks{Ui}d

i=1 and middle

{Di}d
blocks M

Input: Caching Interval N , Branch Index m
Input: Output from step xt, timestep t
Output: predicted output at t − N step
▷ 1. Cache Step - Calculate ϵθ (xt, t) and xt−1
h0 ← xt
for i = 1, . . . , d do
hi ← Di(hi−1)

▷ hi for down-sampling features

ud+1 ← M (hd)
for i = d, . . . , 1 do

if i = m then

▷ ui for up-sampling features

Store ui+1 in Cache
ui ← Ui(Concat(ui+1, hi))
(cid:17)

(cid:16)

u1

xt − 1−αt√
xt−1 = 1√
+ σtz
αt
1− ¯αt
▷ 2. Retrieve Step - Calculate {xt−i}N
for n = 2, . . . , N do
h0 ← xt−n+1
for i = 1, . . . , m do
hi ← Di(hi−1)
Retrieve ui+1 from Cache
for i = m, . . . , 1 do

i=2

▷ z ∼ N (0, I)

ui ← Ui(Concat(ui+1, hi))
(cid:17)

(cid:16)

xt−n = 1√
αt

xt − 1−αt√
1− ¯αt

u1

return xt−N

+ σtz ▷ z ∼ N (0, I)

B. Varying Hyper-parameters in Non-uniform

1:N Strategy

In the non-uniform 1:N strategy, the two hyper-parameters
involved are the center c and the power p, which is used to
determine the sequence of timesteps for conducting the en-
tire model inference. We test on LDM-4-G for the impact of
these hyper-parameters. Results are shown in Table 7 and
Table 8. From these two tables, a clear trend is evident in
the observations: as the parameters p and c are incremented,
there is an initial improvement in the generated image qual-
ity followed by a subsequent decline. This pattern affirms
the effectiveness of the strategy and also aligns with the lo-

12

cation of the significant decrease in similarity observed in
Figure 2(c).

Center

FID ↓

c = 10
c = 20
c = 50
c = 80
c = 100
c = 120
c = 150
c = 200

8.26
8.17
7.77
7.36
7.16
7.11
7.33
8.09

ImageNet 256 × 256
sFID ↓

IS ↑

Precision ↑ Recall ↑

8.47
8.46
8.16
7.76
7.51
7.34
7.36
7.79

160.3
161.18
163.74
166.21
167.52
167.85
166.04
160.50

75.69
75.77
76.23
76.93
77.30
77.44
77.09
75.85

48.93
48.95
49.18
49.75
49.64
50.08
49.98
49.69

Table 7. Varying Center c with the power p equals to 1.2. Here the
caching interval is set to 20.

Power

FID ↓

p = 1.05
p = 1.1
p = 1.2
p = 1.3
p = 1.4
p = 1.5

7.36
7.25
7.11
7.09
7.13
7.25

ImageNet 256 × 256
sFID ↓

IS ↑

Precision ↑ Recall ↑

7.52
7.44
7.34
7.35
7.39
7.44

166.12
166.82
167.85
167.97
167.68
166.82

77.06
77.17
77.44
77.56
77.42
77.17

50.38
50.13
50.08
50.34
50.26
50.13

Table 8. Varying Power p with the center c equals to 120. Here the
caching interval is also set to 20.

N=2 N=3 N=5 N=10 N=20

Center - c
Power - p

120
1.2

120
1.2

110
1.4

110
1.2

120
1.4

Table 9. Hyper-parameters for the non-uniform 1:N strategy in
LDM-4-G

N=2 N=3 N=4 N=5 N=6 N=7 N=8

Center - c
Power - p

Center - c
Power - p

15
1.5

20
1.3

15
1.3

20
1.4

15
1.4

20
1.4

10
1.5

15
1.3

15
1.3

15
1.5

15
1.4

15
1.5

10
1.4

20
1.3

Table 10. Hyper-parameters for the non-uniform 1:N strategy in
Stable Diffusion v1.5.

Selected Hyper-parameters for non-uniform 1:N For
experiments in LDM, the optimal hyper-parameters and
shown in Table 9. For experiments in Stable Diffusion, we
chose center timesteps from the set {0, 5, 10, 15, 20, 25}
and power values from the set {1.1, 1.2, 1.3, 1.4, 1.5, 1.6}.

ImageNet 256 × 256 (250 DDIM Steps)

Method

FID ↓

sFID ↓

IS ↑

Precision ↑ Recall ↑ Method

FID ↓

sFID ↓

IS ↑

Precision ↑ Recall ↑

Baseline - LDM-4*

Uniform - N=2
Uniform - N=3
Uniform - N=5
Uniform - N=10
Uniform - N=20

3.37

3.39
3.44
3.59
4.41
8.23

5.14

5.11
5.11
5.16
5.57
8.08

204.56

204.09
202.79
200.45
191.11
161.83

82.71

82.75
82.65
82.36
81.26
75.31

53.86

54.07
53.81
53.31
51.53
50.57

Baseline - LDM-4

Non-uniform - N=2
Non-uniform - N=3
Non-uniform - N=5
Non-uniform - N=10
Non-uniform - N=20

3.60

3.46
3.49
3.63
4.27
7.36

-

5.14
5.13
5.12
5.42
7.76

247.67

204.12
203.22
200.04
193.11
166.21

87.0

83.21
83.18
83.07
81.75
76.93

48.0

53.53
53.44
53.25
51.84
49.75

Table 11. Comparing non-uniform and uniform 1:N strategy in class-conditional generation for ImageNet using LDM-4-G. *We regenerate
the images using the official checkpoint of LDM-4-G.

The optimal hyper-parameter values employed in our exper-
iments are detailed in Table 10.

From the selected hyper-parameters, we found out that
the optimal values vary slightly across different datasets. A
noticeable trend is observed, indicating that the majority of
optimal parameters tend to center around the 15th timestep,
accompanied by a power value of approximately 1.4.

C. Non-uniform 1:N v.s. Uniform 1:N

We have shown the comparison of the non-uniform 1:N ver-
sus uniform 1:N strategy on Stable Diffusion in Figure 6.
Here, we extend the comparison to ImageNet with LDM-4-
G, and the corresponding results are detailed in Table 11.

In accordance with the observations from Table 11, a
consistent pattern emerges compared to the findings on
Stable Diffusion. Notably, when employing a substantial
caching interval, the non-uniform strategy demonstrates a
notable improvement, with the FID increasing from 8.23
to 7.36 with N=20. However, when dealing with a smaller
caching interval (N<5), the strategy does not yield an en-
hancement in image quality. In fact, in certain cases, it may
even lead to a slight degradation of images, as evidenced by
the FID increasing from 3.39 to 3.46 for N=2.

D. Varying Skip Branches

In Table 12, we show the impact on image quality as we
vary the skip branch for executing DeepCache. For our ex-
periments, we employ the uniform 1:N strategy with N=5,
and the sampling of DDIM still takes 100 steps. From
the results in the table, we observe that the choice of skip
branch introduces a trade-off between speed and image fi-
delity. Specifically, opting for the first skip branch with no
down-sampling blocks and one up-sampling block yields
approximately 3× acceleration, accompanied by a reduc-
tion in FID to 7.14. Additionally, certain skip branches ex-
hibit significant performance variations, particularly the 6-
th branch. The results emphasize an extra trade-off between
speed and image quality, complementing the earlier noted
trade-off linked to different sampling steps. This particular
trade-off operates at the level of model size granularity and
can be achieved without incurring additional costs.

CIFAR-10 32 × 32

Skip Branch MACs ↓ Throughput ↑

Speed ↑

FID ↓

1
2
3
4
5
6
7
8
9
10
11
12

1.60G
2.24G
3.01G
3.89G
4.58G
5.31G
5.45G
5.60G
5.88G
5.95G
5.99G
6.03G

29.60
22.24
18.11
15.44
13.15
11.46
11.27
11.07
10.82
10.73
10.67
10.59

3.023×
2.272×
1.850×
1.577×
1.343×
1.171×
1.151×
1.131×
1.105×
1.096×
1.089×
1.082×

7.14
5.94
5.73
5.69
5.51
4.93
4.92
4.76
4.54
4.57
4.52
4.48

Table 12. Effect of different skip branches. Here we test the impact
under the uniform 1:5 strategy.

E. Prompts

Prompts in Figure 1(a):
• A bustling city street under the shine of a full moon
• A picture of a snowy mountain peak, illuminated by the

first light of dawn

• dark room with volumetric light god rays shining through

window onto stone fireplace in front of cloth couch

• A photo of an astronaut on a moon
• A digital illustration of a medieval town, 4k, detailed,

trending in artstation, fantasy

• A photo of a cat. Focus light and create sharp, defined

edges
Prompts in Figure 5:

• A person in a helmet is riding a skateboard
• There are three vases made of clay on a table
• A very thick pizza is on a plate with one piece taken.
• A bicycle is standing next to a bed in a room.
• A kitten that is sitting down by a door.
• A serene mountain landscape with a flowing river, lush
greenery, and a backdrop of snow-capped peaks, in the
style of an oil painting.

• A delicate floral arrangement with soft, pastel colors and
light, flowing brushstrokes typical of watercolor paint-
ings.

• A magical winter wonderland at night. Envision a land-

13

Steps

Throughput

Speed CLIP Score N Throughput

Speed Uniform 1:N Non-Uniform 1:N

PLMS

DeepCache

50
45
40
35
30
25
20
15

0.230
0.251
0.307
0.333
0.384
0.470
0.538
0.664

1.00
1.09
1.34
1.45
1.67
2.04
2.34
2.89

29.51
29.40
29.35
29.24
29.24
29.32
29.15
28.58

1
2
3
4
5
6
7
8

-
0.333
0.396
0.462
0.494
0.529
0.555
0.582

-
1.45
1.72
2.01
2.15
2.30
2.41
2.53

-
29.54
29.50
29.53
29.41
29.30
29.11
28.97

-
29.51
29.59
29.57
29.50
29.46
29.42
29.26

Table 13. Stable Diffusion v1.5 on PartiPrompt

Steps

Throughput

Speed CLIP Score N Throughput

Speed Uniform 1:N Non-Uniform 1:N

PLMS

DeepCache

50
45
40
35
30
25
20
15

0.237
0.252
0.306
0.352
0.384
0.453
0.526
0.614

1.00
1.06
1.29
1.49
1.62
1.91
2.22
2.59

30.24
30.14
30.19
30.09
30.04
29.99
29.82
29.39

1
2
3
4
5
6
7
8

-
0.356
0.397
0.448
0.500
0.524
0.555
0.583

-
1.50
1.68
1.89
2.11
2.21
2.34
2.46

-
30.31
30.33
30.28
30.19
30.04
29.90
29.76

-
30.37
30.34
30.31
30.23
30.18
30.10
30.02

Table 14. Stable Diffusion v1.5 on MS-COCO 2017

Figure 8. DDPM for LSUN-Churches: Samples with DDIM-100 steps (upper line) and DDIM-100 steps + DeepCache with N=5 (lower
line). The speedup Ratio here is 1.85×.

scape covered in fresh snow, with twinkling stars above,
a cozy cabin with smoke rising from its chimney, and a
gentle glow from lanterns hung on the trees

• A photograph of an abandoned house at the edge of a for-
est, with lights mysteriously glowing from the windows,
set against a backdrop of a stormy sky. high quality pho-
tography, Canon EOS R3.

• A man holding a surfboard walking on a beach next to the

ocean.

F. Detailed Results for Stable Diffusion

We furnish the elaborate results corresponding to Figure 6
in Table 13 and Table 14. Given the absence of a defini-
tive N for aligning the throughput of PLMS, we opt for an

14

alternative approach by exploring results for various N val-
ues. Additionally, we assess the performance of the PLMS
algorithm across different steps. Analyzing the data from
these tables reveals that for N < 5, there is minimal varia-
tion in the content of the image, accompanied by only slight
fluctuations in the CLIP Score.

G. More Samples for Each Dataset

We provide the generated images for each model and each
dataset in Figure 8, Figure 9, Figure 10, Figure 11 and Fig-
ure 12.

OriginalOriginalOursOursFigure 9. Stable Diffusion v1.5: Samples with 50 PLMS steps (upper line) and 50 PLMS steps + DeepCache with N=5 (lower line). The
speedup Ratio here is 2.15×. Here we select prompts from the MS-COCO 2017 validation set.

15

Figure 10. LDM-4-G for ImageNet: Samples with DDIM-250 steps (upper line) and DDIM-250 steps + DeepCache with N=10 (lower
line). The speedup Ratio here is 6.96×.

16

Figure 11. DDPM for LSUN-Bedroom: Samples with DDIM-100 steps (upper line) and DDIM-100 steps + DeepCache with N=5 (lower
line). The speedup Ratio here is 1.48×.

17

Figure 12. DDPM for LSUN-Churches: Samples with DDIM-100 steps (upper line) and DDIM-100 steps + DeepCache with N=5 (lower
line). The speedup Ratio here is 1.48×.

18
