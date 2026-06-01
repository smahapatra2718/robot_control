"""
IK with a rest-pose bias toward a seed configuration.

Same as `solve_ik` but adds a small `rest_cost` pulling the solution toward
`q_seed`. This breaks ties in IK's null space so the solver returns the
closest valid configuration to where the robot already is, instead of an
arbitrary distant IK branch.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_seeded(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    q_seed: onp.ndarray,
    rest_weight: float = 2.0,
) -> onp.ndarray:
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    assert q_seed.shape == (robot.joints.num_actuated_joints,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_seeded_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        jnp.array(q_seed),
        jnp.array(rest_weight),
    )
    return onp.array(cfg)


@jdc.jit
def _solve_ik_seeded_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    q_seed: jax.Array,
    rest_weight: jax.Array,
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
        pk.costs.rest_cost(joint_var, q_seed, weight=rest_weight),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=jaxls.VarValues.make([joint_var.with_value(q_seed)]),
        )
    )
    return sol[joint_var]
