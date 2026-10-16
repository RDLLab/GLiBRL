"""KL divergences between GLiBRL task beliefs.

GLiBRL keeps a conjugate belief over the linear weights of the current task. For the
transition model the belief is a Matrix-Normal-Inverse-Wishart over ``W`` in
``R^{k x s}`` (``k`` latent features, ``s`` output dims),

    Sigma ~ IW(Omega, nu),      W | Sigma ~ MN(M, Sigma (x) Xi^-1),

and the reward belief is the same family with ``s = 1``. When ``noise_mode`` is
``"diag"`` / ``"scalar"`` the Inverse-Wishart is replaced by ``s`` (resp. one)
independent inverse-gammas with shape ``nu / 2`` and rate ``Omega / 2`` -- the
parameterisation :meth:`GLiBRL.task_posterior_inner` maintains. When
``full_bayesian`` is off, ``Omega`` is a *fixed* noise precision rather than a
random scale, so the noise block contributes nothing to the divergence.

Everything below expects the ``[num_tasks, 1]`` leading batch dims of ``TaskTuple``
and returns arrays of that shape.
"""

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jaxtyping import Array, Float

from metaworld_algorithms.types import TaskParam, TaskTuple


def _mv_lgamma(a: Float[Array, "..."], dim: int) -> Float[Array, "..."]:
    """Log of the ``dim``-variate gamma function."""
    offsets = 0.5 * jnp.arange(dim, dtype=a.dtype)
    return dim * (dim - 1) / 4 * jnp.log(jnp.pi) + gammaln(a[..., None] - offsets).sum(-1)


def _mv_digamma(a: Float[Array, "..."], dim: int) -> Float[Array, "..."]:
    """``dim``-variate digamma function."""
    offsets = 0.5 * jnp.arange(dim, dtype=a.dtype)
    return digamma(a[..., None] - offsets).sum(-1)


def _inverse_wishart_kl(
    omega_q: TaskParam, nu_q: TaskParam, omega_p: TaskParam, nu_p: TaskParam, dim: int
) -> Float[Array, "..."]:
    """KL( IW(omega_q, nu_q) || IW(omega_p, nu_p) ) for ``dim x dim`` scale matrices."""
    logdet_q = jnp.linalg.slogdet(omega_q)[1]
    logdet_p = jnp.linalg.slogdet(omega_p)[1]
    trace_term = jnp.trace(
        jnp.linalg.solve(omega_q, omega_p), axis1=-2, axis2=-1
    )
    return (
        0.5 * nu_p * (logdet_q - logdet_p)
        + 0.5 * (nu_q - nu_p) * _mv_digamma(0.5 * nu_q, dim)
        + _mv_lgamma(0.5 * nu_p, dim)
        - _mv_lgamma(0.5 * nu_q, dim)
        + 0.5 * nu_q * (trace_term - dim)
    )


def _inverse_gamma_kl(
    omega_q: TaskParam, nu_q: TaskParam, omega_p: TaskParam, nu_p: TaskParam
) -> Float[Array, "..."]:
    """KL between inverse-gammas with shape ``nu / 2`` and rate ``omega / 2``.

    ``omega_*`` carry one entry per independent output dim in their last axis; the
    per-dim divergences are summed, since the dims are independent given the weights.
    """
    per_dim = (
        0.5 * nu_p * (jnp.log(omega_q) - jnp.log(omega_p))
        + 0.5 * (nu_q - nu_p) * digamma(0.5 * nu_q)
        + gammaln(0.5 * nu_p)
        - gammaln(0.5 * nu_q)
        + 0.5 * nu_q * (omega_p / omega_q - 1.0)
    )
    return per_dim.sum(-1)


def belief_kl_terms(
    posterior: tuple[TaskParam, TaskParam, TaskParam, TaskParam],
    prior: tuple[TaskParam, TaskParam, TaskParam, TaskParam],
    out_dim: int,
    noise_mode: str = "full",
    full_bayesian: bool = True,
) -> dict[str, Float[Array, "..."]]:
    """The three additive terms of KL( posterior || prior ) for one belief block.

    Both beliefs are ``(mean, xi, omega, nu)`` as laid out by
    :meth:`GLiBRL.task_posterior_inner`. The divergence factorises as

        KL(noise) + E_{Sigma ~ posterior}[ KL( MN(m_q, Sigma (x) xi_q^-1)
                                            || MN(m_p, Sigma (x) xi_p^-1) ) ],

    and the inner expectation only needs ``E[Sigma^-1]``, which is ``nu Omega^-1``
    for both the Inverse-Wishart and the inverse-gamma parameterisations. It splits
    further into a term that depends only on the precisions and one that measures
    the mean shift, so the total is reported as three parts:

    ``precision``
        How much tighter the posterior's weight covariance is. Grows like
        ``0.5 * out_dim * latent_dim * ln N`` as data accumulates.
    ``mean``
        The mean shift ``m_q - m_p``, measured in units of the *prior* covariance
        and scaled by ``E[Sigma^-1]``. Against a unit-scale prior this is roughly
        ``0.5 * ||m_q||^2 / sigma^2``, so it dominates once the model fits well;
        against a sharp prior (another posterior) it is amplified by that prior's
        precision.
    ``noise``
        Movement of the noise-covariance belief alone. Grows like ``0.5 * ln N``
        per independent variance parameter, and is zero when ``full_bayesian``
        is off.

    Args:
        posterior: The belief the divergence is taken with respect to.
        prior: The reference belief.
        out_dim: Number of output dims ``s`` the weights map into.
        noise_mode: ``"full"``, ``"diag"`` or ``"scalar"``, matching the algorithm's.
        full_bayesian: Whether the noise covariance is itself part of the belief.

    Returns:
        The ``precision``, ``mean`` and ``noise`` terms, each with the beliefs'
        batch shape. Their sum is the divergence.
    """
    m_q, xi_q, omega_q, nu_q = posterior
    m_p, xi_p, omega_p, nu_p = prior

    latent_dim = m_q.shape[-2]

    # Weight block: the E_Sigma[.] above splits into a Sigma-free precision term ...
    xi_trace = jnp.trace(jnp.linalg.solve(xi_q, xi_p), axis1=-2, axis2=-1)
    weight_kl = 0.5 * out_dim * (
        xi_trace
        - latent_dim
        + jnp.linalg.slogdet(xi_q)[1]
        - jnp.linalg.slogdet(xi_p)[1]
    )

    # ... and a mean-shift term weighted by E[Sigma^-1].
    delta = m_q - m_p
    quad = jnp.swapaxes(delta, -1, -2) @ xi_p @ delta  # [..., out_dim, out_dim]

    # The dof / shape parameter lives in a 1x1 matrix slot.
    nu_q_scalar = nu_q[..., 0, 0]
    nu_p_scalar = nu_p[..., 0, 0]

    if not full_bayesian:
        # Omega is a fixed noise precision here, and identical in both beliefs.
        mean_kl = 0.5 * jnp.trace(omega_q @ quad, axis1=-2, axis2=-1)
        noise_kl = jnp.zeros_like(mean_kl)
    elif noise_mode == "full":
        mean_kl = 0.5 * nu_q_scalar * jnp.trace(
            jnp.linalg.solve(omega_q, quad), axis1=-2, axis2=-1
        )
        noise_kl = _inverse_wishart_kl(
            omega_q, nu_q_scalar, omega_p, nu_p_scalar, out_dim
        )
    elif noise_mode == "diag":
        omega_q_diag = omega_q[..., 0, :]
        omega_p_diag = omega_p[..., 0, :]
        mean_kl = 0.5 * nu_q_scalar * (
            jnp.diagonal(quad, axis1=-2, axis2=-1) / omega_q_diag
        ).sum(-1)
        noise_kl = _inverse_gamma_kl(
            omega_q_diag,
            nu_q_scalar[..., None],
            omega_p_diag,
            nu_p_scalar[..., None],
        )
    elif noise_mode == "scalar":
        omega_q_scalar = omega_q[..., 0, 0]
        omega_p_scalar = omega_p[..., 0, 0]
        mean_kl = (
            0.5
            * nu_q_scalar
            / omega_q_scalar
            * jnp.trace(quad, axis1=-2, axis2=-1)
        )
        noise_kl = _inverse_gamma_kl(
            omega_q_scalar[..., None],
            nu_q_scalar[..., None],
            omega_p_scalar[..., None],
            nu_p_scalar[..., None],
        )
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    return {"precision": weight_kl, "mean": mean_kl, "noise": noise_kl}


def belief_kl(
    posterior: tuple[TaskParam, TaskParam, TaskParam, TaskParam],
    prior: tuple[TaskParam, TaskParam, TaskParam, TaskParam],
    out_dim: int,
    noise_mode: str = "full",
    full_bayesian: bool = True,
) -> Float[Array, "..."]:
    """KL( posterior || prior ) for one weight belief block.

    See :func:`belief_kl_terms`, which this sums.
    """
    terms = belief_kl_terms(
        posterior, prior, out_dim=out_dim, noise_mode=noise_mode,
        full_bayesian=full_bayesian,
    )
    return terms["precision"] + terms["mean"] + terms["noise"]


def task_belief_kl_terms(
    posterior: TaskTuple,
    prior: TaskTuple,
    state_dim: int,
    noise_mode: str = "full",
    full_bayesian: bool = True,
) -> dict[str, Float[Array, "..."]]:
    """KL( posterior || prior ) of a whole task belief, split by block and term.

    Args:
        posterior: The belief the divergence is taken with respect to.
        prior: The reference belief.
        state_dim: The observation dimension the transition weights map into.
        noise_mode: ``"full"``, ``"diag"`` or ``"scalar"``, matching the algorithm's.
        full_bayesian: Whether the noise covariance is itself part of the belief.

    Returns:
        ``transition`` and ``reward`` divergences, plus each block's ``precision``,
        ``mean`` and ``noise`` terms under a ``t_`` / ``r_`` prefix.
    """
    mt_q, xit_q, _, omegat_q, nut_q, mr_q, xir_q, _, omegar_q, nur_q = posterior
    mt_p, xit_p, _, omegat_p, nut_p, mr_p, xir_p, _, omegar_p, nur_p = prior

    transition = belief_kl_terms(
        (mt_q, xit_q, omegat_q, nut_q),
        (mt_p, xit_p, omegat_p, nut_p),
        out_dim=state_dim,
        noise_mode=noise_mode,
        full_bayesian=full_bayesian,
    )
    # The reward is scalar-valued, so its noise belief is always a single
    # inverse-gamma regardless of the transition model's noise_mode.
    reward = belief_kl_terms(
        (mr_q, xir_q, omegar_q, nur_q),
        (mr_p, xir_p, omegar_p, nur_p),
        out_dim=1,
        noise_mode="full",
        full_bayesian=full_bayesian,
    )

    terms = {f"t_{name}": value for name, value in transition.items()}
    terms |= {f"r_{name}": value for name, value in reward.items()}
    terms["transition"] = sum(transition.values())
    terms["reward"] = sum(reward.values())
    return terms


def task_belief_kl(
    posterior: TaskTuple,
    prior: TaskTuple,
    state_dim: int,
    noise_mode: str = "full",
    full_bayesian: bool = True,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """KL( posterior || prior ) of a whole task belief, split by block.

    See :func:`task_belief_kl_terms` for the per-term breakdown.

    Returns:
        The transition-belief and reward-belief divergences.
    """
    terms = task_belief_kl_terms(
        posterior, prior, state_dim=state_dim, noise_mode=noise_mode,
        full_bayesian=full_bayesian,
    )
    return terms["transition"], terms["reward"]
