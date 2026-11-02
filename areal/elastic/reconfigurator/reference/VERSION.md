# Reference Reconfigurator

Unified Evaluator with independent per-forecast state, explicit Hot/Cold execution,
Actor supply and memory checks, and independent role history.

Component costs are organized as Actor, Rollout and publication models. Backend rules
reuse shared operators; profile/calibration loading is separate from queries. Actor
remains resident after initialization. Basic v1 and numeric profiles are preserved.
