# Methods

## Training signals

For each question, the current student samples four responses. The reward is 1 when the final numerical answer is valid and correct, and 0 otherwise. Each batch contains four questions.

**RL.** Rewards are centered and divided by their population standard deviation within each question:

$$
A_i = \frac{r_i - \bar r}{\operatorname{std}(r) + 10^{-8}}.
$$

When all four rewards agree, their advantages are zero. The response advantage is applied to every generated token, including EOS when present. This is a GRPO-style group-relative policy-gradient objective with no value model or separate reference-policy penalty.

**OPD.** The frozen teacher scores the student's sampled tokens. At token position $t$, the detached learning signal is

$$
D_{i,t} = \log p_{\mathrm{teacher}}(y_{i,t}\mid x,y_{i,<t})
          - \log p_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t}).
$$

Both models score the same prefix and token. This token-local surrogate provides teacher feedback along student-generated responses. It does not compute a full-vocabulary distillation loss or the exact full-sequence reverse-KL gradient.

**Joint training.** The token signal is $A_i + D_{i,t}$, with coefficient 1 for each term. Sequential methods apply one signal at a time and retain Adam's state when switching.

## Update

The loss uses the probability ratio between the current and rollout student:

$$
\rho_{i,t} = \exp\bigl(\log p_\theta(y_{i,t})-\log p_{\mathrm{old}}(y_{i,t})\bigr),
$$

$$
L = -\frac{1}{B}\sum_{i=1}^{B}\frac{1}{T_i}\sum_{t=1}^{T_i}
\min\bigl(\rho_{i,t}S_{i,t},\operatorname{clip}(\rho_{i,t},0.8,1.2)S_{i,t}\bigr).
$$

Here $S$ is the RL, OPD, or joint signal; $B=16$ responses. Each response receives equal weight after averaging over its completion tokens. Prompt and padding tokens are excluded.

There is one optimizer update per fresh batch, accumulated over individual responses. Current and old token log probabilities are checked before the update. Their ratio starts near one, so clipping is normally inactive in this one-step setting. Batches with entirely zero signals skip the optimizer step, including Adam's moment updates.

Base and teacher weights use BF16; LoRA master weights and scored log probabilities use FP32. Only the student adapter is trained. The teacher and base student weights remain fixed.

## Time allocation

Every run has a 7,200-second upper training allowance. A fixed 480-second guard stops new batches near the limit. Sequential methods switch at the first completed batch reaching 3,600 charged seconds; the realized first-phase share is about 53.5–54.8% of consumed time.

The clock includes rollout generation, reward calculation, student and teacher scoring, optimization, and within-batch evidence and checkpoint work. Initialization, final export, evaluation, and separate audits are reported outside that allowance.

## Evaluation and uncertainty

The final adapter is selected by the time rule. Evaluation uses greedy decoding, the same answer format, and the frozen reward parser. Incorrect, invalid, and truncated answers remain in the denominator.

Mean accuracy gives equal weight to the three training seeds. The paired bootstrap draws three seed labels with replacement and independently draws 1,319 question indices with replacement. Every method uses the same draws. The initial baseline uses the corresponding question draw in each resample.

The reported intervals are the 2.5th and 97.5th percentiles of 20,000 paired differences, using RNG seed 20261003. They describe variation across the observed training seeds and test questions, conditional on the shared initialization. They are unadjusted for multiple comparisons.
