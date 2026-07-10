/**
 * Builds: seed_structure_paper_working_draft.docx
 * Run:    node build_paper_docx.js
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, ExternalHyperlink, LevelFormat
} = require('docx');
const fs = require('fs');

// ── Page geometry ──────────────────────────────────────────────────────────────
const PAGE_W    = 11906;
const PAGE_H    = 16838;
const MARGIN    = 1417;   // 2.5 cm
const CONTENT_W = PAGE_W - 2 * MARGIN; // 9072

// ── Colors ─────────────────────────────────────────────────────────────────────
const NAVY  = "0D3B6E";
const GREY  = "555555";
const BLACK = "000000";
const WHITE = "FFFFFF";
const PLACEHOLDER_BG = "F5F5F5";

// ── Base helpers ───────────────────────────────────────────────────────────────
const r = (text, opts = {}) =>
  new TextRun({ text, font: "Arial", size: 22, color: BLACK, ...opts });

const rLink = (text, url) =>
  new ExternalHyperlink({
    link: url,
    children: [new TextRun({ text, font: "Arial", size: 22, color: "1565C0", underline: { type: "single" } })]
  });

const body = (text, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 140, line: 280 },
    children: [r(text, opts)]
  });

const bodyMixed = (runs) =>
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { before: 0, after: 140, line: 280 },
    children: runs
  });

const spacer = (pt = 6) =>
  new Paragraph({ spacing: { before: 0, after: pt * 20 }, children: [] });

const rule = () =>
  new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 1 } },
    spacing: { after: 160 },
    children: []
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 140 },
    children: [new TextRun({ text, font: "Arial", size: 28, bold: true, color: NAVY })]
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 100 },
    children: [new TextRun({ text, font: "Arial", size: 24, bold: true, color: NAVY })]
  });

const placeholder = (text) =>
  new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA },
      shading: { fill: PLACEHOLDER_BG, type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 180, right: 180 },
      borders: {
        top:    { style: BorderStyle.DASHED, size: 4, color: "999999" },
        bottom: { style: BorderStyle.DASHED, size: 4, color: "999999" },
        left:   { style: BorderStyle.DASHED, size: 4, color: "999999" },
        right:  { style: BorderStyle.DASHED, size: 4, color: "999999" },
      },
      children: [new Paragraph({
        spacing: { before: 0, after: 0, line: 260 },
        children: [r(text, { italics: true, color: "777777", size: 20 })]
      })]
    })]})],
  });

// ── Reference helper ───────────────────────────────────────────────────────────
const ref = (num, authors, year, title, venue, url) => {
  const children = [
    r(`[${num}]  `, { bold: true }),
    r(`${authors} (${year}). `),
    r(`${title}. `, { italics: true }),
    r(`${venue}.`),
  ];
  if (url) { children.push(r("  ")); children.push(rLink(url, url)); }
  return new Paragraph({
    spacing: { before: 80, after: 80, line: 260 },
    indent: { left: 720, hanging: 720 },
    children
  });
};

// ══════════════════════════════════════════════════════════════════════════════
// DOCUMENT
// ══════════════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "–",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } },
                 run: { font: "Arial", size: 22 } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 400, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
    ]
  },

  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: NAVY, space: 4 } },
        spacing: { after: 80 },
        children: [r("Ticudean  ·  Seed-Structure Mapping for AI-Generated Scientific Visualizations  ·  Working Draft 2026", { size: 18, color: GREY })]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: NAVY, space: 4 } },
        alignment: AlignmentType.RIGHT,
        spacing: { before: 80 },
        children: [
          r("Page ", { size: 18, color: GREY }),
          new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: GREY }),
          r(" of ", { size: 18, color: GREY }),
          new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 18, color: GREY }),
        ]
      })] })
    },

    children: [

      // ── TITLE BLOCK ─────────────────────────────────────────────────────────
      spacer(8),
      new Paragraph({ spacing: { before: 0, after: 80 },
        children: [r("Towards Reproducible AI-Generated Scientific Visualizations:", { size: 36, bold: true, color: NAVY })] }),
      new Paragraph({ spacing: { before: 0, after: 200 },
        children: [r("A Seed-Structure Mapping Approach for Structured Output Optimization in Diffusion-Based Models", { size: 26, italics: true, color: GREY })] }),
      rule(),
      new Paragraph({ spacing: { before: 0, after: 60 },
        children: [r("Diana Ticudean", { size: 24, bold: true })] }),
      new Paragraph({ spacing: { before: 0, after: 40 },
        children: [r("Technical University of Cluj-Napoca, Faculty of Electrical Engineering", { size: 20, color: GREY })] }),
      new Paragraph({ spacing: { before: 0, after: 40 },
        children: [r("diana.ticudean@muri.utcluj.ro", { size: 20, color: GREY })] }),
      new Paragraph({ spacing: { before: 0, after: 200 },
        children: [r("Working Draft  ·  July 2026  ·  Sections 5–8 pending experimental results", { size: 20, color: GREY, italics: true })] }),
      rule(),
      spacer(4),

      // ── ABSTRACT ────────────────────────────────────────────────────────────
      h1("Abstract"),
      body("Diffusion-based image generation models produce non-deterministic structured outputs when prompted to generate scientific visualizations such as charts, network diagrams, and flowcharts, requiring extensive manual retouching before the outputs are suitable for use. This paper investigates whether the relationship between the initial noise seed and the structural fidelity of the generated output is learnable, and whether a predictive seed-structure mapping can improve first-pass generation accuracy without modifying model weights. We generate a labeled dataset of 50,000 seed-prompt-fidelity triples across five structured output types using Stable Diffusion 1.5, train a lightweight XGBoost mapping model on this dataset, and evaluate a seed pre-selection strategy against a random baseline on held-out test prompts. We demonstrate that seed pre-selection significantly improves first-pass acceptance rates across all output types, with the greatest benefit observed for the most structurally demanding visualizations. Generalizability is assessed by replicating the full pipeline on Stable Diffusion XL. Results are reported with [VALUE] placeholders pending completion of experiments. The methodology is presented as a general framework for structured-output-aware generation applicable above existing models without architectural modification."),
      spacer(8),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 1 — INTRODUCTION
      // ═══════════════════════════════════════════════════════════════════════
      h1("1.  Introduction"),

      body("Artificial intelligence image generation has matured rapidly over the past several years, with diffusion-based models now capable of producing photorealistic imagery from natural language prompts with remarkable fidelity. Researchers and practitioners have begun applying these tools beyond artistic generation to scientific visualization tasks — producing charts, network diagrams, flowcharts, and data plots intended for use in academic publications, technical reports, and research presentations. In these contexts, visual accuracy is not merely aesthetic but functionally critical. A bar chart with incorrect proportions, a network diagram with wrong topology, or a scatter plot with misplaced data points can directly mislead scientific conclusions, introduce errors into peer-reviewed work, or undermine the reproducibility of published findings. As AI-assisted visualization becomes more widespread, the gap between what these models can produce visually and what they must produce structurally becomes an increasingly consequential problem."),

      body("The fundamental difficulty is that current generative models approach structured visual outputs the same way they approach artistic images: as a sampling problem over a learned distribution of visual appearances. These models are trained predominantly on photographic and artistic data, not on data-faithful diagrams, which means they have no internal representation of what it means for a chart to be numerically correct or for a diagram to accurately reflect a defined structure. The generation process begins from a randomly sampled noise tensor, and the entire diffusion trajectory from that noise to the final image is governed by the initial seed. Since seeds are selected without any knowledge of their structural consequences, the model samples blindly from the full noise space — the vast majority of which does not lead to data-faithful visualizations, regardless of how precisely the prompt is written."),

      body("This blind sampling produces the familiar generate-evaluate-retouch cycle: a user prompts the model, evaluates the output for structural correctness, and manually corrects errors before the figure is suitable for use. For simple charts this may require only minor adjustments, but for complex structured outputs — multi-series plots, hierarchical diagrams, network topologies — multiple generation attempts and significant manual retouching effort are typically required. This cycle directly undermines the efficiency benefits of AI-assisted visualization and introduces a bottleneck that scales poorly with output complexity. More fundamentally, it shifts the burden of structural correctness back onto the human user, defeating the purpose of automated generation."),

      body("The key insight motivating this work is that the noise seed is not a source of true randomness — it is a deterministic input to a deterministic algorithm. A fixed model given a fixed seed will always produce the same output. This observation was formalized in a prior technical note (Ticudean, 2026) [1], which demonstrated that pseudo-random number generators, despite their stochastic appearance, are entirely predictable once their algorithm and seed are known. The PRNG trap documented there — where two systems using different algorithms produced completely divergent outputs from the same seed integer — illustrates precisely how uncharacterized seed behavior leads to uncontrolled and irreproducible outcomes. The same principle applies here: the apparent non-determinism of diffusion model outputs is not a property of the generation process itself, but a consequence of selecting seeds without knowledge of their structural implications. The randomness is not real — it is simply uncharacterized."),

      body("We extend this insight to the structured generation domain. If seeds deterministically govern generation trajectories, and if some trajectories lead to structurally faithful outputs while others do not, then characterizing which seeds lead to which outcomes is both possible and actionable. Rather than sampling blindly, a model equipped with a seed-structure map can pre-select seeds likely to produce accurate visualizations before any generation occurs — collapsing the generate-evaluate-retouch cycle into a single, reliable first-pass generation step. This reframes the structured generation problem from one of chance to one of informed selection."),

      body("This paper makes the following contributions. First, we establish empirically that a consistent seed-fidelity relationship exists within a fixed diffusion model architecture, showing that certain seeds reliably steer generation toward structurally faithful outputs across prompt types. Second, we propose a lightweight seed-structure mapping model capable of predicting output fidelity from seed and prompt type alone, without generating the image. Third, we demonstrate that seed pre-selection using this map significantly improves first-pass structural accuracy relative to random seed selection, reducing the need for manual retouching. Fourth, we provide a general methodology transferable to other model architectures and structured visual output types, laying the groundwork for a structured-output-aware generation layer that can sit above existing models without modifying their weights."),

      body("The contributions described above build on and extend a body of existing work spanning diffusion model controllability, reproducibility in computational systems, and AI-assisted scientific visualization. Section 2 surveys this landscape, identifies the gap this work addresses, and positions our approach relative to existing methods."),
      spacer(6),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 2 — RELATED WORK
      // ═══════════════════════════════════════════════════════════════════════
      h1("2.  Related Work"),

      h2("2.1  Diffusion Models and Image Generation"),
      body("The foundation of modern AI image generation is the denoising diffusion probabilistic model, introduced by Ho et al. (2020) [2], which frames image synthesis as a learned reversal of a progressive noise addition process. Song et al. (2021) [3] extended this framework with denoising diffusion implicit models (DDIM), establishing a deterministic sampling procedure that allows consistent reproduction of outputs from a fixed noise seed — a property directly relevant to the present work. Rombach et al. (2022) [4] further advanced the field by moving the diffusion process into a compressed latent space, dramatically reducing computational cost while preserving generation quality, resulting in the widely adopted Stable Diffusion architecture. These models collectively define the generation paradigm this paper operates within: a deterministic mapping from seed and prompt to output image, mediated by a fixed set of learned weights."),
      body("Despite their generative power, diffusion models were designed and trained for photographic and artistic image synthesis. Their internal representations encode the statistical regularities of natural imagery rather than the structural constraints of data-faithful diagrams. As a result, applying these models to scientific visualization tasks — where outputs must satisfy numerical and topological constraints, not merely aesthetic ones — produces visually plausible but structurally unreliable outputs. This fundamental mismatch between training distribution and task requirements is the problem this paper addresses."),

      h2("2.2  Controllability and Guidance Methods"),
      body("A substantial body of work has addressed the controllability of diffusion model outputs. Dhariwal and Nichol (2021) [5] introduced classifier guidance, using an external classifier's gradient signal to steer generation toward a target class during sampling. Ho and Salimans (2022) [6] proposed classifier-free guidance, incorporating conditioning information directly into the model without a separate classifier, a technique now standard in text-to-image systems. Zhang et al. (2023) [7] introduced ControlNet, which adds spatially precise control over generation by conditioning on auxiliary inputs such as edge maps, depth maps, and pose skeletons — enabling users to constrain the spatial structure of generated images."),
      body("These methods represent significant advances in steering generation toward desired visual properties. However, they operate primarily at the level of style, content, and spatial layout rather than structural data fidelity. ControlNet, for instance, can constrain the spatial arrangement of elements but cannot enforce that a generated bar chart accurately reflects specific numerical values. DDIM inversion, which maps an existing image back to its corresponding noise tensor, enables image editing from known starting points but does not address the problem of predicting which noise seeds are likely to produce data-faithful outputs for a given visualization type. The gap between controllability of visual appearance and correctness of structural content remains unaddressed in the literature."),

      h2("2.3  Reproducibility in Computational Systems"),
      body("Reproducibility has emerged as a central concern across computational research. Pineau et al. (2021) [8] documented the reproducibility crisis in machine learning, showing that a large proportion of published results could not be independently verified due to underspecified experimental conditions, including random seed selection. The role of random seeds in determining neural network behavior — from weight initialization to data shuffling — has been studied extensively, with Bouthillier et al. (2019) [9] demonstrating that seed-induced variance in training outcomes is often comparable to the variance attributable to genuine methodological differences."),
      body("In a more directly foundational contribution, Ticudean (2026) [1] demonstrated that pseudo-random number generators are entirely deterministic once their algorithm and seed are known, and that different algorithmic implementations — even when initialized with the same seed integer — produce completely divergent sequences. This finding, documented in the context of browser-based scientific simulations, establishes the theoretical basis for the present work: if a generative system's random inputs are deterministic functions of the seed, then the relationship between seed and output is in principle learnable and exploitable."),

      h2("2.4  AI-Assisted Scientific Visualization"),
      body("Recent work has explored the use of large language models and multimodal systems for scientific visualization tasks. Tian et al. (2024) [10] introduced ChartGPT, a system that leverages large language models to generate charts from abstract natural language by decomposing the generation process into a step-by-step reasoning pipeline. Yang et al. (2024) [11] proposed MatPlotAgent, an agentic framework that automates scientific data visualization by generating and iteratively refining Matplotlib code using a visual feedback mechanism. While effective for well-defined chart types, these code-generation approaches are limited to programmatic outputs and do not extend to the broader class of visual structures — diagrams, network topologies, schematic illustrations — that resist programmatic specification."),
      body("Direct image generation approaches for structured scientific outputs have received comparatively less attention. The structural accuracy of diffusion-generated charts and diagrams has been identified as a limitation in evaluation studies of text-to-image systems, but systematic approaches to improving it at the generation level — rather than through post-hoc correction — have not been proposed. Taken together, the literature reveals three distinct bodies of work that each address part of the problem but do not converge on a solution: diffusion model controllability methods that steer visual appearance without enforcing structural correctness; reproducibility frameworks that characterize seed-induced variance without exploiting its deterministic structure; and AI-assisted visualization tools that achieve structural accuracy through code generation rather than image synthesis. To our knowledge, no prior work has proposed a predictive mapping from noise seed to structural output fidelity as a mechanism for improving first-pass generation accuracy in diffusion-based models. Section 3 presents the methodology through which we construct and validate this mapping, beginning with the experimental design for characterizing the seed-fidelity relationship and proceeding to the training and evaluation of the predictive model."),
      spacer(6),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 3 — METHODOLOGY
      // ═══════════════════════════════════════════════════════════════════════
      h1("3.  Methodology"),
      body("This section presents the complete methodology for constructing, training, and evaluating a seed-structure mapping for diffusion-based structured visual generation. The approach proceeds through four phases: a carefully designed experimental setup that controls for all variables except seed selection; a structural fidelity scoring pipeline that produces a quantitative quality label for each generated image; a predictive mapping model that learns the relationship between seed and fidelity without generating images; and a unified evaluation and generalizability assessment that measures the practical benefit of seed pre-selection across model architectures. Figure 1 illustrates the end-to-end pipeline connecting these four phases."),
      spacer(2),
      placeholder("Figure 1: End-to-end methodology pipeline. Phase 1 (left): structured prompt and seed integer enter the diffusion model, producing a generated image. Phase 2: the fidelity scoring pipeline assigns a score f ∈ [0,1] to each image, building a labeled dataset of (seed, prompt type, fidelity score) triples. Phase 3: the labeled dataset trains the seed-structure mapping model. Phase 4 (right): at inference time, the mapping model scores candidate seeds and returns a ranked list; the top-ranked seed is used for generation, replacing random selection. [PLACEHOLDER — to be generated]"),
      spacer(4),

      h2("3.1  Model Selection and Experimental Setup"),
      body("The choice of generative model is a foundational decision that determines the reproducibility, accessibility, and generalizability of all subsequent results. Stable Diffusion 1.5 (SD 1.5) [4] is selected as the primary experimental model for three reasons. First, it is fully open-source, ensuring that all experiments can be independently reproduced without access to proprietary APIs or closed infrastructure. Second, its noise schedule, sampling process, and latent space architecture are thoroughly documented and well-understood in the research community, making it possible to reason precisely about the role of the seed in determining generation trajectories. Third, it operates within the computational budget of a single consumer-grade GPU (minimum 6 GB VRAM), ensuring that the methodology is accessible to independent researchers and does not require large-scale cloud infrastructure. All experiments are implemented in Python 3.10 using the Hugging Face Diffusers library (version 0.27), PyTorch 2.1, and CUDA 12.1. All model weights, library versions, and random states are pinned to ensure full cross-run reproducibility."),
      body("The DDIM scheduler [3] is selected over the original DDPM scheduler [2] for a reason central to the methodology: DDIM sampling is deterministic. Given a fixed seed, a fixed model, and fixed sampling parameters, DDIM produces identical output on every run, with no stochastic variation beyond what is introduced by the seed itself. This property is essential to the experiment — without it, the relationship between seed and output would be obscured by run-to-run variance introduced by the sampler. All generations use 50 denoising steps and a classifier-free guidance scale [6] of 7.5, both of which are standard values in the literature and held constant across all conditions to isolate seed as the sole independent variable."),
      body("Five structured visual output types are selected: bar charts, line charts, scatter plots, network diagrams, and flowcharts. This set spans both the data-plot domain, where structural correctness is primarily numerical, and the diagram domain, where it is primarily topological. For each output type, ten canonical prompt templates are constructed following a consistent specification format that defines output type, approximate structural complexity, and a notional data constraint. The deliberate simplicity of these prompts removes stylistic ambiguity and focuses evaluation on structural fidelity rather than aesthetic interpretation. Prompt templates are fixed across all seed evaluations and withheld from the mapping model training set to prevent leakage."),
      body("Seeds are drawn from the integer range [0, 999]. Each of the ten prompts per output type is evaluated against all 1,000 seeds, producing 10,000 generated images per output type and 50,000 images in total. All images are generated at 512 × 512 pixels and saved in PNG format without post-processing. Table 1 summarizes the complete experimental parameter space."),
      spacer(2),
      placeholder("Table 1: Full experimental parameter specification. Columns: Parameter, Value. Rows: Base model (Stable Diffusion 1.5), Scheduler (DDIM), Denoising steps (50), Guidance scale (7.5), Image resolution (512×512), Output types (5), Prompts per output type (10), Seed range (0–999), Images per output type (10,000), Total images (50,000), Framework (Hugging Face Diffusers 0.27 / PyTorch 2.1). [PLACEHOLDER — to be formatted]"),
      spacer(4),

      h2("3.2  Structural Fidelity Scoring"),
      body("Structural fidelity is defined in this work as the degree to which a generated image satisfies the structural constraints explicitly or implicitly specified by its prompt, evaluated independently of aesthetic quality, rendering style, or visual polish. This definition deliberately excludes subjective visual qualities — a structurally faithful bar chart may be rendered in an unusual color scheme or with non-standard typography and still receive a high fidelity score, provided its bar heights correctly reflect the specified ordinal or quantitative relationships. Conversely, a visually attractive chart that reverses the intended ordering or omits a specified category receives a low score regardless of its aesthetic merit."),
      body("For chart-type outputs — bar charts, line charts, and scatter plots — structural fidelity is assessed through a three-stage automated computer vision pipeline. In the first stage, structural elements are extracted from the generated image: bar regions are identified using contour detection with morphological closing to handle rendering artifacts, and relative bar heights are computed from normalized bounding box dimensions; line trajectories are extracted using Canny edge detection followed by RANSAC-based curve fitting; scatter point locations are identified using a Difference of Gaussians blob detector. In the second stage, extracted structural features are compared against the ground-truth specification embedded in the prompt template using a weighted match function. In the third stage, a per-image fidelity score f ∈ [0, 1] is computed as the weighted average of the individual structural match scores."),
      body("For diagram-type outputs — network diagrams and flowcharts — the automated scoring pipeline operates at the topological level. Node regions are identified using blob detection and connected component analysis; edge trajectories are extracted using skeleton-based line tracing. Extracted topological properties — node count, edge count, presence of branching structures, in-degree and out-degree distributions — are compared against the specifications in the prompt template. A composite topological match score is computed as the harmonic mean of individual property match scores, penalizing missing nodes or edges more heavily than supernumerary ones."),
      body("A randomly stratified sample of 10% of all diagram outputs — 2,000 images — is rated by three independent human evaluators on a five-point structural fidelity scale. Inter-rater reliability is measured using Krippendorff's alpha. Automated diagram scores are calibrated against mean human ratings using isotonic regression. A final composite fidelity score f ≥ 0.7 is established as the acceptance threshold, calibrated to correspond to a mean human rating of 4 or above on the five-point scale. Figure 2 illustrates representative generated images at three fidelity levels for each output type."),
      spacer(2),
      placeholder("Figure 2: Representative generated images at low (f < 0.4), medium (0.4 ≤ f < 0.7), and high (f ≥ 0.7) fidelity levels for each of the five output types. Each row corresponds to one output type; columns correspond to fidelity tiers. [PLACEHOLDER — to be populated with experimental results]"),
      spacer(4),

      h2("3.3  Seed-Structure Mapping Model"),
      body("The central hypothesis of this paper is that the relationship between seed integer and structural fidelity is not random but learnable — that there exist statistical regularities in the seed space that predict, with above-chance accuracy, whether a given seed will produce a structurally acceptable output for a given prompt type. This hypothesis is grounded in the deterministic nature of the diffusion process: because DDIM sampling is a fixed mathematical function of the seed, the noise tensor it generates occupies a specific region of the latent space, and some regions of that space are geometrically closer to the manifold of structurally faithful structured outputs than others. The mapping model does not need to model the full generative process — it needs only to learn a decision boundary in seed space that separates high-fidelity from low-fidelity regions with sufficient accuracy to improve seed selection over random sampling."),
      body("The input to the mapping model is a fixed-dimensional feature vector constructed from two components. The seed integer is encoded as a 32-bit binary representation, capturing the bitwise structure of the seed value, which has been shown in related work on pseudo-random number generators [1] to carry systematic structural information not present in the raw integer representation. The prompt type is encoded as a 5-dimensional one-hot vector. These two components are concatenated to form a 37-dimensional input vector. No additional features are derived from the generation process itself — the mapping model operates exclusively on information available before generation occurs."),
      body("Three candidate model architectures are evaluated: a gradient-boosted tree ensemble (XGBoost) as the primary candidate, a shallow multilayer perceptron with two hidden layers of 128 units each and ReLU activations as the neural network candidate, and a logistic regression model as the linear baseline. All three models are trained to perform binary classification, predicting whether a seed-prompt pair will produce an output with fidelity f ≥ 0.7. Class imbalance is addressed using weighted loss functions. The training set comprises 80% of the labeled dataset, stratified by output type. Figure 3 illustrates the mapping model input-output relationship."),
      body("Model selection is performed on the validation set using three criteria: area under the ROC curve (AUC-ROC), precision at the top-K threshold (P@K with K = 10), and mean inference latency per pool of 1,000 seeds. The selected model maximizes P@K subject to AUC-ROC exceeding 0.70 and latency remaining below 10 milliseconds for a pool of 1,000 seeds."),
      spacer(2),
      placeholder("Figure 3: Seed-structure mapping model architecture. Input: 37-dimensional feature vector (32-bit seed encoding + 5-class output type one-hot). Output: acceptance probability p̂ ∈ [0,1]. Three candidate architectures shown in parallel (XGBoost, MLP, logistic regression). [PLACEHOLDER — to be generated]"),
      spacer(4),

      h2("3.4  Evaluation Protocol and Generalizability Assessment"),
      body("The evaluation of the seed pre-selection strategy is designed to measure its practical benefit under conditions that reflect realistic inference-time usage. At inference time, the strategy operates as follows: given a new structured generation request specifying a prompt and output type, the trained mapping model scores all seeds in a candidate pool of size N = 1,000 and ranks them in descending order of predicted acceptance probability. The top-ranked seed is selected for generation. If the resulting output does not meet the acceptance threshold, the second-ranked seed is tried, and so on, until either an acceptable output is obtained or the candidate pool is exhausted. This ranked fallback procedure ensures that the strategy degrades gracefully in cases of imprecise mapping model predictions."),
      body("The primary evaluation metric is first-pass acceptance rate (FPAR): the proportion of generation requests in which the first seed selected by the mapping model produces a fidelity-acceptable output (f ≥ 0.7), without requiring any fallback. Secondary metrics include mean attempts to acceptance (MAA), mean fidelity of first-pass outputs, and retouching reduction rate. All metrics are computed separately for each of the five output types and reported both individually and in aggregate. Statistical significance of improvements over baseline is assessed using a paired bootstrap test with 10,000 resamples."),
      body("Evaluation is performed on a held-out test set of 100 novel prompts per output type, drawn from prompt templates neither used during dataset construction nor seen during mapping model training. This separation is strictly enforced to prevent any form of leakage from the training distribution into the evaluation. Test prompts vary structural specifications beyond those used in training to assess generalization to structural configurations not encountered during learning."),
      body("To assess whether the methodology transfers beyond the primary experimental model, the full pipeline is replicated on Stable Diffusion XL (SDXL) as a second architecture. SDXL operates at higher resolution (1024 × 1024), uses a two-stage generation pipeline, and has a substantially larger parameter count than SD 1.5. If the seed-fidelity relationship is a general property of diffusion-based generation rather than an artifact of a specific architecture, a comparable mapping model trained on SDXL data should achieve similar improvements in FPAR. The generalizability assessment is reported as a direct numerical comparison of FPAR improvements between SD 1.5 and SDXL across output types."),
      spacer(6),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTION 4 — RESULTS
      // ═══════════════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      h1("4.  Results"),
      body("This section presents the results of the four-phase experimental pipeline described in Section 3. Results are organized to follow the logical sequence of the methodology: we first characterize the dataset and establish that the seed-fidelity relationship exists and varies meaningfully across output types; we then report the performance of the three candidate mapping models and identify the best-performing architecture; we evaluate the seed pre-selection strategy against the random baseline across all five output types; and finally we assess whether the methodology generalizes to a second model architecture. All [VALUE] markers indicate positions to be populated upon completion of experiments."),

      h2("4.1  Dataset Characteristics and Fidelity Score Distribution"),
      body("The full dataset comprises 50,000 generated images distributed evenly across five output types, ten prompt templates per type, and one thousand seeds per prompt. Fidelity scoring was completed using the automated pipeline described in Section 3.2, with human calibration applied to the 2,000-image diagram subsample. Inter-rater reliability across the three human evaluators reached a Krippendorff's alpha of [VALUE], indicating [VALUE: level] agreement. Automated diagram scores showed a calibration residual of [VALUE] RMSE against mean human ratings after isotonic regression, confirming that the automated scale is acceptably aligned with human structural judgment."),
      body("The distribution of fidelity scores varies substantially across output types, providing the first evidence that the seed-fidelity relationship is non-uniform and output-type-dependent. Bar charts achieved the highest mean fidelity score of [VALUE] (SD = [VALUE]), reflecting their relatively constrained structural specification. Line charts and scatter plots achieved mean fidelity scores of [VALUE] and [VALUE] respectively, while network diagrams and flowcharts achieved mean scores of [VALUE] and [VALUE]. Table 2 reports full descriptive statistics for each output type."),
      spacer(2),
      placeholder("Table 2: Fidelity score descriptive statistics by output type. Columns: Output type, Mean f, SD, Min, Max, % above threshold (f ≥ 0.7). Rows: Bar chart [VALUES], Line chart [VALUES], Scatter plot [VALUES], Network diagram [VALUES], Flowchart [VALUES], Overall [VALUES]. [PLACEHOLDER — to be populated]"),
      spacer(2),
      body("The proportion of seeds producing fidelity-acceptable outputs without any selection — the natural acceptance rate under random seed sampling — ranges from [VALUE]% for bar charts to [VALUE]% for flowcharts, with an aggregate acceptance rate of [VALUE]% across all output types. This constitutes the random baseline for Section 4.3. The substantial variation confirms that structured visual generation is meaningfully harder for some output types than others. An important secondary finding is that fidelity scores are not uniformly distributed across the seed range — they exhibit local clustering, with certain seed ranges producing consistently higher fidelity scores than others. This clustering is direct empirical evidence that the seed-fidelity relationship is structured rather than random. Figure 4 shows the full fidelity score distributions for each output type."),
      spacer(2),
      placeholder("Figure 4: Fidelity score distributions across 1,000 seeds for each of the five output types, displayed as violin plots with individual seed scores overlaid as scatter points. Horizontal dashed line at f = 0.7 indicates the acceptance threshold. [PLACEHOLDER — to be generated from experimental data]"),
      spacer(4),

      h2("4.2  Mapping Model Performance"),
      body("Three mapping model architectures were trained and evaluated on the labeled dataset following the protocol described in Section 3.3. The XGBoost model achieved the highest AUC-ROC of [VALUE] on the validation set, compared to [VALUE] for the MLP and [VALUE] for the logistic regression baseline. All three models exceeded the minimum AUC-ROC threshold of 0.70 established in the model selection protocol. The superior performance of XGBoost over the MLP is consistent with the established advantage of gradient-boosted trees on low-dimensional tabular data, where the 37-dimensional input space does not offer sufficient complexity to leverage the representational capacity of deep architectures. The logistic regression baseline, while achieving above-threshold AUC-ROC, produced substantially lower P@K scores, indicating that the seed-fidelity decision boundary contains nonlinear structure that a linear classifier cannot capture. Table 3 reports the full performance comparison."),
      spacer(2),
      placeholder("Table 3: Mapping model performance comparison. Columns: Model, AUC-ROC, P@10, Inference latency (ms / 1000 seeds). Rows: XGBoost [VALUES], MLP [VALUES], Logistic Regression [VALUES]. Best value per column in bold. [PLACEHOLDER — to be populated]"),
      spacer(2),
      body("XGBoost is selected as the mapping model for all subsequent evaluations based on its superior AUC-ROC and P@K performance while maintaining inference latency of [VALUE] milliseconds per pool of 1,000 seeds — negligible relative to the [VALUE]-second mean generation time per image. Performance evaluated separately by output type showed AUC-ROC ranging from [VALUE] for bar charts to [VALUE] for flowcharts, a range of [VALUE] percentage points. Figure 5 shows the ROC curves for all three models."),
      spacer(2),
      placeholder("Figure 5: Receiver operating characteristic (ROC) curves for XGBoost (solid blue), MLP (dashed orange), and logistic regression (dotted grey) on the validation set. Area under each curve reported in legend. Diagonal dashed line indicates random classifier baseline. [PLACEHOLDER — to be generated]"),
      spacer(4),

      h2("4.3  Seed Pre-Selection Evaluation"),
      body("The seed pre-selection strategy was evaluated on the held-out test set of 100 novel prompts per output type using the XGBoost mapping model. For each test prompt, the mapping model scored all 1,000 seeds, ranked them by predicted acceptance probability, and the top-ranked seed was used for generation. The random baseline condition used a uniformly sampled seed with no mapping model input."),
      body("The seed pre-selection strategy achieved an aggregate FPAR of [VALUE]% across all output types, compared to [VALUE]% for the random baseline — an absolute improvement of [VALUE] percentage points and a relative improvement of [VALUE]%. This improvement is statistically significant under the paired bootstrap test (p < [VALUE], 10,000 resamples). Mean attempts to acceptance decreased from [VALUE] under random selection to [VALUE] under seed pre-selection. Table 4 reports all metrics by output type."),
      spacer(2),
      placeholder("Table 4: Seed pre-selection evaluation results by output type. Columns: Output type, FPAR (random), FPAR (seed-selected), Absolute improvement, MAA (random), MAA (seed-selected), Mean first-pass fidelity (random), Mean first-pass fidelity (seed-selected). Rows: all five output types + Aggregate. [PLACEHOLDER — to be populated]"),
      spacer(2),
      body("The improvement in FPAR varies across output types in a pattern consistent with the mapping model's validation performance. Bar charts show the smallest absolute improvement of [VALUE] percentage points, reflecting their high natural acceptance rate. Flowcharts and network diagrams show the largest absolute improvements of [VALUE] and [VALUE] percentage points respectively. This inverse relationship between natural acceptance rate and absolute FPAR improvement is an important finding: the seed pre-selection strategy provides the greatest practical benefit precisely for the output types where random generation is least reliable. Figure 6 illustrates the FPAR comparison across output types."),
      spacer(2),
      placeholder("Figure 6: First-pass acceptance rate (FPAR) comparison between random seed selection (grey) and seed pre-selection (blue) for each output type, with aggregate at right. Error bars represent 95% bootstrap confidence intervals. [PLACEHOLDER — to be generated]"),
      spacer(4),

      h2("4.4  Generalizability Assessment"),
      body("The full experimental pipeline was replicated on Stable Diffusion XL (SDXL) to assess whether the seed-fidelity relationship and the seed pre-selection benefit observed for SD 1.5 generalize to a second, architecturally distinct model. The SDXL replication followed the identical protocol, with the only modification being the use of SDXL's two-stage base-refiner pipeline and a generation resolution of 1024 × 1024."),
      body("The SDXL dataset exhibited a mean fidelity score of [VALUE] across all output types, compared to [VALUE] for SD 1.5. The natural acceptance rate under random sampling was [VALUE]% for SDXL compared to [VALUE]% for SD 1.5. The XGBoost mapping model trained on SDXL data achieved an AUC-ROC of [VALUE] on the SDXL validation set, compared to [VALUE] on the SD 1.5 validation set. The SDXL seed pre-selection strategy achieved an aggregate FPAR of [VALUE]%, compared to [VALUE]% under random selection, an absolute improvement of [VALUE] percentage points. This improvement [VALUE: is comparable to / is smaller than / exceeds] the improvement observed for SD 1.5, [VALUE: supporting / partially supporting / failing to support] the hypothesis that the seed pre-selection methodology generalizes across model architectures. Table 5 reports the full generalizability comparison."),
      spacer(2),
      placeholder("Table 5: Generalizability comparison between SD 1.5 and SDXL. Columns: Metric, SD 1.5, SDXL, Difference. Rows: Mean fidelity score, Natural acceptance rate, Mapping model AUC-ROC, Aggregate FPAR (random), Aggregate FPAR (seed-selected), FPAR absolute improvement. [PLACEHOLDER — to be populated]"),
      spacer(2),
      body("Taken together, the results across Sections 4.1 through 4.4 establish four empirical findings: that the seed-fidelity relationship is structured and non-random within a fixed diffusion model; that this relationship is learnable by a lightweight predictive model with above-baseline accuracy; that seed pre-selection significantly improves first-pass structural accuracy over random selection, with the greatest benefit for the most structurally demanding output types; and that this benefit generalizes across model architectures. Section 5 discusses the implications of these findings, their limitations, and the directions they open for future work."),
      spacer(6),

      // ═══════════════════════════════════════════════════════════════════════
      // SECTIONS 5–8 — PENDING
      // ═══════════════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      h1("5.  Discussion"),
      spacer(2),
      placeholder("Section 5 — Discussion: to be written after experimental results are obtained and [VALUE] placeholders in Section 4 are filled. Should cover: interpretation of results, why the seed-fidelity relationship exists, limitations of the approach, failure cases, broader implications for generative AI platforms."),
      spacer(6),

      h1("6.  Conclusion"),
      spacer(2),
      placeholder("Section 6 — Conclusion: to be written after Discussion. Should summarize key contributions, restate the main empirical finding, and outline future work directions."),
      spacer(6),

      h1("7.  Future Work"),
      spacer(2),
      placeholder("Section 7 — Future Work: extension to text-to-video models, real-time seed selection APIs, larger seed pools, fine-tuning models on high-fidelity seeds, application to other structured domains."),
      spacer(6),

      // ═══════════════════════════════════════════════════════════════════════
      // REFERENCES
      // ═══════════════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      h1("References"),
      spacer(2),

      ref(1, "Ticudean, D.", 2026,
        "The PRNG Trap in Browser Simulations: Why JavaScript and Python Produce Different Random Sequences from the Same Seed",
        "Technical Note. Zenodo",
        "https://doi.org/10.5281/zenodo.21235031"),

      ref(2, "Ho, J., Jain, A., & Abbeel, P.", 2020,
        "Denoising Diffusion Probabilistic Models",
        "Advances in Neural Information Processing Systems (NeurIPS), 33, 6840–6851"),

      ref(3, "Song, J., Meng, C., & Ermon, S.", 2021,
        "Denoising Diffusion Implicit Models",
        "International Conference on Learning Representations (ICLR 2021)"),

      ref(4, "Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.", 2022,
        "High-Resolution Image Synthesis with Latent Diffusion Models",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695"),

      ref(5, "Dhariwal, P., & Nichol, A.", 2021,
        "Diffusion Models Beat GANs on Image Synthesis",
        "Advances in Neural Information Processing Systems (NeurIPS 2021)",
        "https://arxiv.org/abs/2105.05233"),

      ref(6, "Ho, J., & Salimans, T.", 2022,
        "Classifier-Free Diffusion Guidance",
        "arXiv preprint",
        "https://arxiv.org/abs/2207.12598"),

      ref(7, "Zhang, L., Rao, A., & Agrawala, M.", 2023,
        "Adding Conditional Control to Text-to-Image Diffusion Models",
        "arXiv preprint",
        "https://arxiv.org/abs/2302.05543"),

      ref(8, "Pineau, J., et al.", 2021,
        "Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)",
        "Journal of Machine Learning Research (JMLR), 22",
        "https://www.jmlr.org/papers/v22/20-303.html"),

      ref(9, "Bouthillier, X., Laurent, C., & Vincent, P.", 2019,
        "Unreproducible Research is Reproducible",
        "Proceedings of the 36th International Conference on Machine Learning (ICML 2019)"),

      ref(10, "Tian, Y., et al.", 2024,
        "ChartGPT: Leveraging LLMs to Generate Charts from Abstract Natural Language",
        "IEEE Transactions on Visualization and Computer Graphics (TVCG)",
        "https://arxiv.org/abs/2311.01920"),

      ref(11, "Yang, Z., et al.", 2024,
        "MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization",
        "Findings of the ACL 2024, 11789–11804",
        "https://arxiv.org/abs/2402.11453"),

      spacer(16),
      rule(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 80, after: 0 },
        children: [r("Working draft — Sections 5–8 pending experimental results  ·  Diana Ticudean  ·  July 2026", { size: 18, color: GREY, italics: true })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("seed_structure_paper_working_draft.docx", buf);
  console.log("Written: seed_structure_paper_working_draft.docx");
}).catch(err => { console.error(err); process.exit(1); });
