from transformers import TrainerCallback

class LossStoppingCallback(TrainerCallback):
    """
    Stop training when loss down to a certain value. In order to save training time.
    
    """
    def __init__(self,
                 loss_threshold: float = 2.5,
                 start_epoch: int = 3,
                 ):
        """
        Args:
            loss_threshold (float): The threshold of loss.  
            start_epoch (int): The epoch to start the loss checking. 
        """
        self.loss_threshold = loss_threshold
        self.start_epoch = start_epoch
        
    def on_evaluate(self, args, state, control, **kwargs):
        loss_value = state.log_history[-2]["loss"]
        epoch = state.log_history[-1]["epoch"]
        if loss_value < self.loss_threshold and epoch >= self.start_epoch:
            control.should_training_stop = True