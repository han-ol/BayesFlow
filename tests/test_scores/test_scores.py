import keras
import pytest


def test_require_argument_k():
    from bayesflow.scores import NormedDifferenceScore

    with pytest.raises(TypeError) as excinfo:
        NormedDifferenceScore()

    assert "missing 1 required positional argument: 'k'" in str(excinfo)


def test_score_output(scoring_rule, random_conditions):
    if random_conditions is None:
        random_conditions = keras.ops.convert_to_tensor([[1.0]])

    scoring_rule.set_head_shapes_from_target_shape(random_conditions.shape)
    print(scoring_rule.get_config())
    estimates = {
        k: scoring_rule.get_link(k)(keras.random.normal((random_conditions.shape[0],) + head_shape))
        for k, head_shape in scoring_rule.head_shapes.items()
    }
    score = scoring_rule.score(estimates, random_conditions)

    assert score.ndim == 0


def test_mean_score_optimality(mean_score, random_conditions):
    if random_conditions is None:
        random_conditions = keras.ops.convert_to_tensor([[1.0]])

    mean_score.set_head_shapes_from_target_shape(random_conditions.shape)
    key = "value"
    suboptimal_estimates = {key: keras.random.uniform(random_conditions.shape)}
    optimal_estimates = {key: random_conditions}

    suboptimal_score = mean_score.score(suboptimal_estimates, random_conditions)
    optimal_score = mean_score.score(optimal_estimates, random_conditions)

    assert suboptimal_score > optimal_score
    assert keras.ops.isclose(optimal_score, 0)
