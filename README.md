# Critic-Guided Self-Adaptation for Multilingual Continual Learning


Multilingual question answering has advanced with larger models and broader supervision, yet practical adaptation still struggles to balance diversity and faithfulness. When we fine-tune on a mixture of languages, the model often learns phrasing and style that are too close to the training distribution and fails to generalize to new wording or domains. Attempts to inject variety through back-translation, paraphrasing, or format changes tend to help only when the edits remain faithful to the original meaning. Without clear control on faithfulness, edited data can drift away from the actual answer, which reduces exact-match accuracy and introduces unexpected gaps across languages. This tension between diversity and fidelity motivates our study.

We ask a narrow, controlled question: can we improve multilingual and cross-lingual Question-Answering by retraining on the original TyDiQA examples with critic-approved self-edits that introduce measured semantic diversity while preserving meaning? Our setting is simple. We work only with TyDiQA and always answer in the query's native language. We compare three training conditions: (i) the original TyDiQA pairs (*no_edits*); (ii) self-edits restricted to QA-style paraphrases where a critic selects one edit per example (*qa_only*); and (iii) self-edits drawn from several formats such as QA, rewrite, implications, and chain of thought, which are filtered by the critic to keep a single safe variant per example (*all_formats*). The goal is to isolate the effect of controlled diversity, rather than scale, architecture, or training.

Formally, let the multilingual supervision be

$$\mathcal{D} = \{(c_i^{(l)}, q_i^{(l)}, y_i^{(l)})\}_{i=1}^{N_l}, l \in \mathcal{L}$$

where $c$ is the context, $q$ is the question, $y$ is the actual answer span, and $l$ indexes language. For each triplet we generate $K$ candidate edits with format $f \in \{\text{qa, rewrite, implications, cot}\}$:

$$E_{i,k}^{(l,f)} = \text{EditGen}(c_i^{(l)}, q_i^{(l)}, y_i^{(l)}, f), \quad k = 1, \ldots, K.$$

We compute a drift score $d(E) \in [0, 1]$ from cross-lingual embeddings and keep only edits in a target band $\mathcal{B}$ that encourages diversity without departing from the original meaning. A critic $\kappa(\cdot)$ then scores faithfulness and quality and selects one edit per example:

$$\hat{E}_i^{(l,f)} = \arg \max_{k, d(E_{i,k}^{(l,f)}) \in \mathcal{B}} \kappa(E_{i,k}^{(l,f)}).$$

From these pieces we assemble three training sets under identical token budgets:

$$\mathcal{T}^{\text{no}} = \mathcal{D}, \quad \mathcal{T}^{\text{qa}} = \{(c_i^{(l)}, q_i^{(l)}, \hat{E}_i^{(l,\text{qa})})\}, \quad \mathcal{T}^{\text{all}} = \{(c_i^{(l)}, q_i^{(l)}, \hat{E}_i^{(l,f)})\}_{f \in \{\text{qa,rewrite,implications,cot}\}}$$

We fine-tune a small instruction model $A_0$ with LoRA on each set to obtain $A^{\text{no}}, A^{\text{qa}}, A^{\text{all}}$. At inference time the model maps a native-language query to a native-language answer,

$$f_A : (l, x) \mapsto y^{(l)} \quad \text{with } A \in \{A^{\text{no}}, A^{\text{qa}}, A^{\text{all}}\},$$

and we measure exact match and F1 by language and as macro averages.

We implement this approach by coupling drift-banded self-edit generation with critic gating, then fine-tuning with mixed-language batches. Each example carries its drift and critic scores for auditability, ensuring edits remain faithful while adding diversity. We then compare the three training sets to quantify the gain from critic-guided edits in a controlled setting. We keep the prompt template and decoding fixed across conditions so changes in accuracy trace to the data rather than procedure.
