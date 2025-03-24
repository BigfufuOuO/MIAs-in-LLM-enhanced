from transformers import TrainerCallback

class LossStoppingCallback(TrainerCallback):
    """
    Stop training when loss down to a certain value. In order to save training time.
    
    """
    def __init__(self,
                 loss_threshold: float = 2.5,
                 loss_interval: float = 0.06):
        self.loss_threshold = loss_threshold
        self.loss_interval = loss_interval
        
    def on_evaluate(self, args, state, control, **kwargs):
        loss_value = state.log_history[-2]["loss"]
        if abs(loss_value - self.loss_threshold) < self.loss_interval or loss_value < self.loss_threshold:
            control.should_training_stop = True