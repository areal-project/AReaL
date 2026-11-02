# Basic v1

Basic candidate prediction uses the original aggregate JointSimulator from
`elastic-policy-v1` (`52b16947708b17fecffa1fbbde6a03c3c50d73ed`), with Online
observation/window adaptation. The original batch evaluator and aggregate-rate formula
are retained. It does not call ExpertSimulator or OnlineSimulation.
