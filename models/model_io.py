class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(self, state=None, hidden=None, detection_inputs=None, action_probs=None, high_hidden=None, target=None, last_action=None):
        self.state = state
        self.hidden = hidden
        self.detection_inputs = detection_inputs
        self.action_probs = action_probs

        self.high_hidden = high_hidden
        self.target = target
        
        self.last_action = last_action


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None,x=None, high_hidden=None, high_action=None, termination=None):
        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
        self.x = x

        self.high_hidden = high_hidden
        self.high_action = high_action
        self.termination = termination
