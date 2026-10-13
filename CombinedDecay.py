import tensorflow as tf

class CombinedDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses an exponential decay schedule followed by
    a piecewise constant decay schedule.
    When training a model, it is often useful to lower the learning rate as
    the training progresses. This schedule applies an exponential decay function
    to an optimizer step, given a provided initial learning rate.
    The schedule is a 1-arg callable that produces a decayed learning
    rate when passed the current optimizer step. This can be useful for changing
    the learning rate value across different invocations of optimizer functions.
    
    The exponential decay schedule is computed as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```
    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
      
      The output of the 1-arg function that takes the `step`
      is exponential decay decayed_learning_rate when `step <= boundaries[0]`,
      `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
      and values[-1] when `step > boundaries[-1]`.
      
      For every value in values, value is calculated as follows:
      ```
        values[0] = initial_learning_rate * decay_rate ^ (boundaries[0] / decay_steps)
        values[i+1] = values[i] / values_step[i]
      ```
      where i = 0, ..., n amount of boundaries.
    """

    
    def __init__(self, boundaries, initial_learning_rate, decay_steps, decay_rate, values_steps=2, staircase=False, name=None):
        self.boundaries = boundaries
        self.initial_learning_rate = initial_learning_rate        
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name
        self.values_steps = values_steps

        if isinstance(self.values_steps, list) is not True:
            self.values_steps = [values_steps] * len(boundaries)
        
        self.values = [initial_learning_rate * decay_rate ** (boundaries[0] / decay_steps)]
        for i in range(len(boundaries)):
            self.values.append(self.values[i] / self.values_steps[i])


    def __call__(self, step):
        with tf.name_scope(self.name or "CombinedDecay"):         
            
            def ExponentialDecay():
                initial_learning_rate = tf.convert_to_tensor(
                    self.initial_learning_rate, name="initial_learning_rate"
                )
            
                dtype = initial_learning_rate.dtype
                decay_steps = tf.cast(self.decay_steps, dtype)
                decay_rate = tf.cast(self.decay_rate, dtype)

                global_step_recomp = tf.cast(step, dtype)
                p = global_step_recomp / decay_steps
                if self.staircase:
                    p = tf.floor(p)

                return tf.multiply(initial_learning_rate, tf.pow(decay_rate, p))

            def PiecewiseConstantDecay():
                boundaries = tf.nest.map_structure(
                    tf.convert_to_tensor, tf.nest.flatten(self.boundaries)
                )
                values = tf.nest.map_structure(
                    tf.convert_to_tensor, tf.nest.flatten(self.values)
                )
                x_recomp = tf.convert_to_tensor(step)

                for i, b in enumerate(boundaries):
                    if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
                        # We cast the boundaries to have the same type as the step
                        b = tf.cast(b, x_recomp.dtype.base_dtype)
                        boundaries[i] = b
                pred_fn_pairs = []
                pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
                pred_fn_pairs.append(
                    (x_recomp > boundaries[-1], lambda: values[-1])
                )
                for low, high, v in zip(
                    boundaries[:-1], boundaries[1:], values[1:-1]
                ):
                    # Need to bind v here; can do this with lambda v=v: ...
                    pred = (x_recomp > low) & (x_recomp <= high)
                    pred_fn_pairs.append((pred, lambda v=v: v))

                # The default isn't needed here because our conditions are mutually
                # exclusive and exhaustive, but tf.case requires it.
                default = lambda: values[0]
                return tf.case(pred_fn_pairs, default, exclusive=True)
            
            current_step = tf.convert_to_tensor(step)
            boundary = tf.convert_to_tensor(self.boundaries[0])
            
            if boundary.dtype.base_dtype != current_step.dtype.base_dtype:
                boundary = tf.cast(boundary, current_step.dtype.base_dtype)
            return tf.case([(current_step < boundary, ExponentialDecay)], default=PiecewiseConstantDecay, exclusive=True)
            
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "boundaries": self.boundaries,
            "values": self.values,
            "values_steps": self.values_steps,
            "name": self.name,
        }
