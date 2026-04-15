using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

public class ReacherAgent : Agent
{
    [Header("References")]
    public Transform target;
    public Transform ground;

    [Header("Movement")]
    public float speed = 5f;

    [Header("Spawn Settings")]
    public float yPosition = 0.5f;
    public float minDistanceBetweenAgentAndTarget = 3f;
    public float edgeMargin = 0.5f;

    [Header("Rewards / Termination")]
    public float stepPenalty = -0.001f;
    public float successReward = 1f;
    public float fallPenalty = -1f;
    public float fallYThreshold = -1f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();

        if (rb == null)
        {
            Debug.LogError("ReacherAgent requires a Rigidbody component.");
        }

        if (target == null)
        {
            Debug.LogError("Target is not assigned.");
        }

        if (ground == null)
        {
            Debug.LogError("Ground is not assigned.");
        }
    }

    public override void OnEpisodeBegin()
    {
        ResetEnvironment();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 agentPos = transform.position;
        Vector3 targetPos = target.position;

        sensor.AddObservation(agentPos);
        sensor.AddObservation(targetPos);
        sensor.AddObservation(targetPos - agentPos);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveX = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float moveZ = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        Vector3 move = new Vector3(moveX, 0f, moveZ);
        Vector3 nextPos = rb.position + move * speed * Time.fixedDeltaTime;

        rb.MovePosition(nextPos);

        AddReward(stepPenalty);

        CheckFallCondition();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;

        float moveX = 0f;
        float moveZ = 0f;

        var kb = Keyboard.current;

        if (kb == null)
        {
            actions[0] = 0f;
            actions[1] = 0f;
            return;
        }

        if (kb.tKey.isPressed) moveX += 1f; // right
        if (kb.rKey.isPressed) moveX -= 1f; // left
        if (kb.fKey.isPressed) moveZ += 1f; // forward
        if (kb.sKey.isPressed) moveZ -= 1f; // backward

        actions[0] = moveX;
        actions[1] = moveZ;
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.transform == target)
        {
            AddReward(successReward);
	        Debug.Log("Reach Reward!");
            EndEpisode();
        }
    }

    private void CheckFallCondition()
    {
        if (transform.position.y < fallYThreshold)
        {
            AddReward(fallPenalty);
            EndEpisode();
        }
    }

    private void ResetEnvironment()
    {
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        Vector3 agentPos = GetRandomPosition();
        Vector3 targetPos = GetRandomPositionFarFrom(agentPos);

        rb.position = agentPos;
        target.position = targetPos;
    }

    private Vector3 GetRandomPosition()
    {
        float halfX = 5f * ground.localScale.x - edgeMargin;
        float halfZ = 5f * ground.localScale.z - edgeMargin;

        float x = Random.Range(-halfX, halfX);
        float z = Random.Range(-halfZ, halfZ);

        return new Vector3(
            ground.position.x + x,
            yPosition,
            ground.position.z + z
        );
    }

    private Vector3 GetRandomPositionFarFrom(Vector3 origin)
    {
        for (int i = 0; i < 100; i++)
        {
            Vector3 pos = GetRandomPosition();

            if (Vector3.Distance(origin, pos) >= minDistanceBetweenAgentAndTarget)
            {
                return pos;
            }
        }

        return GetRandomPosition();
    }
}
