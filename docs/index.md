---
layout: page
title: "Long-Context Video Understanding"
---

<!-- MathJax setup -->
<script>
  window.MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
    svg: { fontCache: 'global' }
  };
</script>
<script async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Long-Context Video Understanding

*An agentic approach to understanding multi-hour videos*

## The Problem

Imagine a future where AI lives in the physical world. It collaborates with humans, interacts with nature, understands science, designs experiments, and builds systems. One day, AI will not lie in a computer screen, strapped to a software layer. It may walk the streets, see what we see, and proactively help us solve human problems. That day is within reach, but still far away.

As humans we continually take in visual information, and over long periods of time we recognize trends, pinpoint moments in the past, and understand our personal history. A truly intelligent machine should interact with the same raw visual data humans do so that it understands the systems we've built. The problem of long context video understanding one that we need to solve for the future. It also carries many applications in the moment (understanding medical data, surveillance, video-data analysis, robotics, etc).

This is a problem of computer vision, data representation, long context, and memory. Current ML models support short context lengths of up to $1$ million tokens. That's enough for one highly-compressed hour long video. Even if transformers become fully linearized, the amount of visual data we will collect on day will likely exceed the capacity of online storage. 

So solving long context video understanding requires a few components:
1. An offline representation of a video  
2. Identifying relevant moments in a long video  
3. Strong reasoning over the visual and temporal space

We propose an agentic method to understanding long videos to address each of these components on fully open-source models. We utilize the ReAct framework, which integrates reasoning with tool use. Within this framework, an LLM can both perform step-by-step reasoning and invoke external resources, such as searching through a captions database and communicating with other MLLMs. Prior work such as Deep Video Discovery has explored agentic systems for long-video evaluation, and we build on these ideas and push them further. While reasoning agents are powerful, they remain prone to alignment errors, lossy image-to-text translation, and hallucinations. While experimenting with agents, we also notice their fragility and unreliability, especially on subjective visual data. To address these issues, we introduce a third-party **critic** module that evaluates agent outputs, identifies discrepancies, and prompts re-evaluation. This leads to a novel cycle: *Reason, Act, Critique, React*.

## Related Work

There are a growing number of models being trained for one-shot or few-shot video understanding, where the goal is to directly process video clips and answer questions or generate captions. For instance, models such as LLaVA and VideoGPT extend vision-language pretraining into the temporal domain. Due to the scarcity of large-scale annotated video datasets, these models are typically trained by leveraging existing labeled image datasets alongside smaller annotated video corpora, aligning both modalities in a shared latent space. Fine-tuning on video QA or captioning tasks further enhances their temporal reasoning abilities. However, these approaches remain limited to relatively short clips, due to the computational constraints of short context windows and the cost of modeling long-distance dependencies.

In parallel, Socratic Models highlight how multimodal reasoning can be composed by chaining independently trained models through natural language. This form of agent interaction and orchestration has also been explored at scale in systems like HuggingGPT, where an LLM coordinates specialized models to solve complex multimodal tasks. Our work is inspired by this line of research, but focuses specifically on the long-video setting, where composition is critical given the infeasibility of end-to-end training on multi-hour data.

Recent capabilities have also been boosted by advances in chain-of-thought reasoning where models explicitly generate intermediate reasoning steps (an "internal scratchpad"). This improves both performance on complex multi-step problems but also the planning ability of LLMs, making them better suited for decomposing long-horizon tasks such as multi-hour video understanding.

Finally, long-context chunking and embedding-based retrieval have emerged as practical solutions for handling inputs that exceed model context windows. Dense retrieval methods enable efficient semantic search across large text or video databases, while hierarchical chunking approaches allow models to zoom in from global summaries to local evidence. These ideas directly inform our design, where multi-granularity captions and semantic embeddings form the reasoning space for an LLM agent.

## Method

![Figure 1: Agentic-LV pipeline](fig1.png)

Our task is to solve a set of $n$ questions about video $v$:
$$Q_v \ni \{q_1, q_2, \ldots q_n\}.$$

We begin by procuring a set of three video representations at varying granularity:

1. **Frame-centric captions $C_f$**: We extract frames with their timestamps at 1 FPS, and ask a VLM to caption each with a list of objects, their descriptions, and relationships.  
2. **Character, Event, and Scene captions $C_c$**: Using our frame-centric captions, we log recurring characters, sequences of frames which capture the same event, and sequences of frames within recurring locations and record corresponding timestamps.  
3. **Global summary $C_g$**: Using our frame-centric captions, we curate a global summary with focus on plot, main characters, and general tone of the video.

We keep $C_f$ in a database accessible to the LLM. Captions are written by a VLM, which is prompted to capture significant events, signals, actions, and descriptions of subjects. Each caption is one line to shorten context length and reduce redundancy.

We then embed both sets of captions with an embedding model to enable semantic search.

We equip a reasoning LLM with the following tools:
1. Caption-search function, which embeds prompts, and uses cosine-similarity search to find semantically relevant captions, and their corresponding key frames.  
2. Calls to a vision language model (VLM) which reads frames.

We feed $C_c$ and $C_g$, which are much more compact, to the model at the beginning of each question so that it can extract relevant and specific information for its caption-search queries, and have a general idea of where relevant frame timestamps lie.

Then, we provide a framework for the LLM to systematically answer each question:
1. Parse the question $q_i$, and write down temporal location, setting, subject, and actions to search for.  
2. Using the information from {1}, look through captions $C_f, C_c$ to identify key frames. We allow the model to choose between grep/pattern search, and calls to the semantic caption-finder, and encourage the use/experimentation of both.  
3. For each set of key frames found, query a clip of variable length (determined by the model and question) to the MLLM around the key frames to capture relevant context, OR query the VLM with a set of key frames for specific details.  
4. Repeat steps 2 and/or 3 as many times as necessary to gather all information.

Step 1 allows the model to organize what it's searching for, corresponding to the question.

In step 2, we encourage the LLM to first search through $C_c$, and find relevant frames.

![Figure 2: Trace-viewer layout](fig2.png)

### LLM Organization Practices

We arm the LLM with a scratchpad file to keep a chunkable reference memory as it searches. We also enforce that it records exact clip/frame evidence and reasoning in a file. Ablating with and without the scratchpad show a negligible increase in accuracy.

### Predictability + Critic

We notice a high variance in accuracy across runs with the exact same prompts, captions, and hyperparameters. This is due to the compounding probabilistic nature of autoregressive generative models and the complex and open-ended nature of our back-and-forth task.

We know that our LLMs are capable of complex reasoning paths to solve long video understanding questions because their reasoning traces match the human thought process. However, LLMs can be thrown off at any point in reasoning (wrong key frames, misidentification of subjects, misunderstanding of the question), causing a high variance in accuracy across different seeds.

To mitigate, we test a critic system:

Once our reasoning LLM has decided on a final answer, it outputs a json containing its answer, evidence frames, and reasoning. This is fed to another reasoning critic LLM along with the question and global context. The critic LLM reads the reasoning trace, calls a VLM on the evidence frames, and looks for discrepancies in the frames, holes in the reasoning, or incomplete evidence. It returns a confidence score and suggestions for re-evaluation. Upon passing below a selected threshold (70%), we send the critique to the original LLM for re-evaluation.

The use of a single pass through the critic model increases accuracy by **5.98%**.

We run experiments on a few open source models, with and without scratchpad, with and without a critic model, and on different datasets.

Deepseek V3.1 performs the best on open-source models, reaching an accuracy of **65.18%** with a critic model, **60.19%** without on LVBench. This is a stronger performance than all other open-source model SOTA.

## Token and Cost Analysis

![Figure 1: Token Analysis](full_token_preview.png)

**Context:** a normal hour-long video compresses to about **1,000,000 tokens**. If you naively “pass through the entire video” for every question, you spend ~1M input tokens per question, which is expensive and slow especially if the questions or the video are streamed in, as they are in most real-life cases. 

If instead we have an offline representation that we can requery, we can save on amortized token cost. We precompute multi-granularity captions ($C_f, C_c, C_g$) once, then retrieve only the relevant bits. With the critic enabled, a typical Q&A cycle looks like:

- Per question: about 10,435 input tokens and 3,998 output tokens (combined across VLM and LLM). (This cost on the most expensive open-source models is ~ 3 cents per question)
- Of that, the critic pass** contributes roughly 1,251 input and 575 output tokens (i.e., ~12–13% of the per-question total).

Across **16 questions**, we spend around **166,960 input** and **63,968 output** tokens in total (≈**230,928** tokens overall), which is **~23% of a single 1M-token hour**—and **~96× smaller** than the naïve **16M**-token pass-through.

This creates an amortization effect because the heavy lifting (captions and embeddings) is reused, so the incremental cost of each new question stays near ~10k input tokens + ~4k output tokens*, and we also don't exceed a context window for long videos. As the number of questions grows, the per-question cost collapses toward that small incremental budget instead of scaling linearly with full-video passes. However, we still do have to account for the initial video representation cost.

**Critic tradeoff in practice:**  
We add ~12.7% more tokens per question to achieve ~5.98% increase in absolute accuracy.  
If quality matters, we can also run the critic on an adjustable confidence level for more a more customized accuracy-cost tradeoff.

## Limitations and Discussion

The approach inherits several limitations from captioning-based retrieval. Frame-level captions can be lossy and may ignore subtle but important cues (e.g., identity, small objects, fine-grained actions). Agent reliability is also sensitive to prompt phrasing and decoding stochasticity; although the critic reduces variance, performance can still degrade when the initial key-frame selection is not correct. Moreover, while multi-granularity captions alleviate context constraints, they introduce representation bias: omissions in the captions can cause models to hallucinate findings which don't exist.

Despite these constraints, hierarchical retrieval from $C_c \rightarrow C_f$, combined with the *Reason, Act, Critique, React* loop, provides a useful balance. The method restricts the LLM’s working set to relevant evidence, defers to vision models when necessary, and employs a critic to flag inconsistencies and prompt re-evaluation.

## Conclusion

Long-context video understanding is fundamentally a systems integration problem that requires compact multimodal representations, targeted retrieval, and a strong reasoning cycle. The proposed agent combines hierarchical captions, semantic search, VLM-based image understanding, and a critic to improve robustness. On open-source models, this configuration yields the SOTA score on LVBench by only source models, and results in an approximate 6% absolute accuracy improvement with roughly 13% additional tokens per question. Future work will aim to tighten video representation either by focusing on strengthening the captioner for small-object and identity sensitivity or employing a latnet space repersentation, invoking the critic conditionally based on uncertainty, and enforcing stronger evidence grounding by attaching verifiable frame or clip identifiers to each claim.



