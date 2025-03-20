from transformers import TrainerCallback

class LossStoppingCallback(TrainerCallback):
    """
    Stop training when loss down to a certain value. In order to save training time.
    
    """
    def __init__(self,
                 loss_threshold: float = 2.5):
        self.loss_threshold = loss_threshold
        
    def on_epoch_end(self, args, state, control, **kwargs):
        loss_value = state.log_history[-1]["loss"]
        if loss_value < self.loss_threshold:
            control.should_training_stop = True